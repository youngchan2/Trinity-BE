//! Inspect one S-expression, or a numbered evaluation corpus, without Python/GPU.
use std::{env, error::Error, fs};

use trinity_lowering::analyzer::analyze_text;

fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args().nth(1).ok_or("usage: analyze_ir <IR file>")?;
    let source = fs::read_to_string(path)?;
    let mut valid = 0;
    let mut invalid = 0;
    let mut kernels = 0;
    let mut accesses = 0;
    let numbered = source
        .lines()
        .find(|s| !s.trim().is_empty())
        .is_some_and(|s| {
            s.split_once(':')
                .is_some_and(|(id, _)| id.trim().parse::<usize>().is_ok())
        });
    let entries: Vec<_> = if numbered {
        source
            .lines()
            .filter(|s| !s.trim().is_empty())
            .map(|line| line.split_once(':').ok_or("expected numbered IR entry"))
            .collect::<Result<_, _>>()?
    } else {
        vec![("program", source.as_str())]
    };
    for (id, expression) in entries {
        match analyze_text(expression) {
            Ok(result) => {
                valid += 1;
                kernels += result.kernels().len();
                accesses += result.accesses().len();
                if !numbered {
                    println!("{result:#?}");
                }
            }
            Err(error) => {
                invalid += 1;
                if invalid <= 5 {
                    eprintln!("IR {id}: {error}");
                }
            }
        }
    }
    println!("accepted={valid} rejected={invalid} kernels={kernels} accesses={accesses}");
    if invalid > 0 {
        return Err(
            "some IR entries were rejected (including unresolved dummydata placeholders)".into(),
        );
    }
    Ok(())
}
