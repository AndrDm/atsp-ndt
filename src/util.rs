use indicatif::{ProgressBar, ProgressStyle};
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::time::Duration;

pub fn spinner(msg: &str) -> ProgressBar {
	let sp = ProgressBar::new_spinner();
	sp.set_style(
		ProgressStyle::with_template(
			"{spinner:.green} {msg} - {elapsed_precise}",
		)
		.unwrap()
		.tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
	);
	sp.set_message(msg.to_string());
	sp.enable_steady_tick(Duration::from_millis(100)); //update its animation every 100 ms
	sp
}

/// Load cost matrix from a CSV file with configurable delimiter and a progress spinner.
/// Supports separators: semicolon (;), comma (,), and tab (\t).
/// Supports decimal separators: dot (.) and comma (,), but never both as field and decimal.
pub fn load_cost_matrix(path: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
	let file = File::open(path)?;
	println!("Reading CSV {}", path);

	// Detect field separator from first line
	let mut first_line = String::new();
	{
		let mut buf_reader = BufReader::new(&file);
		use std::io::BufRead;
		buf_reader.read_line(&mut first_line)?;
	}

	// Determine field separator
	let field_sep = if first_line.contains('\t') {
		b'\t'
	} else if first_line.contains(';') {
		b';'
	} else {
		b',' // default to comma
	};

	// Reopen file for actual reading
	let file = File::open(path)?;
	let mut rdr = csv::ReaderBuilder::new()
		.delimiter(field_sep)
		.has_headers(false)
		.from_reader(BufReader::new(file));

	let mut matrix = Vec::new();
	for result in rdr.records() {
		let record = result?;
		let row: Vec<f64> = record
			.iter()
			.map(|x| {
				let mut val = x.trim().to_string();

				// Normalize decimal separator:
				// If field separator is comma, decimal must be dot.
				// Otherwise, replace comma with dot for decimal.
				if field_sep == b',' {
					// Do NOT replace commas (they are field separators)
					// Assume decimals are dots already
				} else {
					// Replace comma with dot for decimal
					val = val.replace(',', ".");
				}

				val.parse::<f64>()
					.unwrap_or_else(|_| panic!("Cannot parse '{}' as f64", x))
			})
			.collect();
		matrix.push(row);
	}

	println!("CSV loaded");
	Ok(matrix)
}
