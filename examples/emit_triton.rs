//! Dependency-free text interface. Tensor metadata uses `NAME dim dim ...`;
//! symbols use `NAME=value`. The library API accepts the same data as Rust maps.
use std::{env, error::Error, fs};
use trinity_lowering::triton::{Options, compile};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("usage: emit_triton IR_FILE SHAPES_FILE OUTPUT.py [NAME=value ...]\nIR_FILE is one expression or a numbered corpus; a corpus writes into OUTPUT directory".into());
    }
    let mut options = Options::default();
    for line in fs::read_to_string(&args[1])?
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
    {
        let mut words = line.split_whitespace();
        let name = words.next().unwrap();
        let shape = words.map(str::parse).collect::<Result<Vec<usize>, _>>()?;
        options.shapes.insert(name.to_owned(), shape);
    }
    for arg in &args[3..] {
        let (name, value) = arg.split_once('=').ok_or("expected NAME=value")?;
        options.symbols.insert(name.into(), value.parse()?);
    }
    let text = fs::read_to_string(&args[0])?;
    if text.trim_start().starts_with('(') {
        fs::write(&args[2], compile(&text, options)?)?;
    } else {
        fs::create_dir_all(&args[2])?;
        let mut accepted = 0;
        let mut rejected = 0;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let (id, ir) = line.split_once(':').ok_or("expected numbered IR")?;
            let id: usize = id.trim().parse()?;
            match compile(ir.trim(), options.clone()) {
                Ok(source) => {
                    fs::write(format!("{}/kernel_{id}.py", args[2]), source)?;
                    accepted += 1;
                }
                Err(error) => {
                    eprintln!("{id}: {error}");
                    rejected += 1;
                }
            }
        }
        println!("generated={accepted}, rejected={rejected}");
        if rejected != 0 {
            return Err("some IR candidates were rejected; see diagnostics".into());
        }
    }
    Ok(())
}
