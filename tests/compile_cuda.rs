//! Opt-in NVCC integration: compilation, device linking and artifact inspection.
//! These tests never load CUDA libraries, prepare a device or execute GPU work.
#[allow(dead_code)]
mod support;

use trinity_lowering::{PhysicalPlan, compile, emit};

fn check(plan: PhysicalPlan) {
    let source = emit(&plan).unwrap();
    let artifact = compile(source).unwrap_or_else(|error| panic!("{error}"));

    assert!(artifact.artifact_path().is_file());
    assert!(artifact.diagnostics().status.unwrap().success());
}

#[test]
#[ignore = "requires CUDA 13.0+ and CUTLASS 4.5.1; does not require a GPU"]
fn streamed_sources_compile_and_link() {
    for k in [64, 128, 192, 320] {
        check(support::gemm(128, 128, k, 1));
    }
    check(support::gemm(256, 256, 192, 1));
    check(support::gemm_chain());

    let mut b = support::Builder::new(1);
    let input = b.input("identity", [128, 128]);
    check(b.finish(input));
}

#[test]
#[ignore = "requires CUDA 13.0+, CUTLASS 4.5.1 and NVSHMEM_HOME at version 3.7.2"]
fn persistent_sources_compile_and_link() {
    check(support::gemm(128, 128, 192, 2));

    let mut b = support::Builder::new(2);
    let input = b.input("identity", [128, 128]);
    check(b.finish(input));

    for backend in ["peer_push", "peer_pull", "one_shot_push_nbi"] {
        for axis in [0, 1] {
            check(support::gather(backend, axis, 2));
            check(support::input_gather(backend, axis, 2, false));
            check(support::lhs_gather(backend, axis, 2));
            check(support::output_gather(backend, axis, 2));
        }
        check(support::input_gather(backend, 0, 2, true));
    }

    for (first, second) in [
        ("peer_push", "peer_pull"),
        ("peer_pull", "peer_push"),
        ("one_shot_push_nbi", "one_shot_push_nbi"),
    ] {
        check(support::peer_chain(first, second, 2));
    }
}

#[test]
#[ignore = "requires CUDA 13.0+ including cuobjdump and NVSHMEM_HOME at version 3.7.2"]
fn persistent_artifact_preserves_nvshmem_module_registration_symbols() {
    for plan in [support::gemm(128, 128, 64, 2), {
        let mut b = support::Builder::new(2);
        let input = b.input("identity", [128, 128]);
        b.finish(input)
    }] {
        let artifact = compile(emit(&plan).unwrap()).unwrap_or_else(|error| panic!("{error}"));
        let cuobjdump = artifact
            .diagnostics()
            .compiler
            .parent()
            .unwrap()
            .join("cuobjdump");
        let output = std::process::Command::new(cuobjdump)
            .arg("--dump-elf-symbols")
            .arg(artifact.artifact_path())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );

        let symbols = String::from_utf8_lossy(&output.stdout);
        // nvshmemx_cumodule_init looks these up by name in the device module.
        for name in ["nvshmemi_device_lib_version_d", "nvshmemi_device_state_d"] {
            assert!(symbols.contains(name), "device linker removed {name}");
        }

        let output = std::process::Command::new("nm")
            .args(["-D", "--defined-only"])
            .arg(artifact.artifact_path())
            .output()
            .unwrap();
        assert!(output.status.success());
        let host_symbols = String::from_utf8_lossy(&output.stdout);
        let names: std::collections::BTreeSet<_> = host_symbols
            .lines()
            .filter_map(|line| line.split_whitespace().last())
            .collect();
        assert_eq!(
            names,
            [
                "trinity_abi",
                "trinity_launch",
                "trinity_prepare",
                "trinity_release",
                "trinity_status"
            ]
            .into_iter()
            .collect()
        );
    }
}
