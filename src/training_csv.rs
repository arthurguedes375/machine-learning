use crate::ml::SetType;
use std::fs::read_to_string;

use na::{DVector};
use csv::{StringRecordsIter};

pub type TrainingData = Vec<Vec<SetType>>;

pub fn clean_csv(records: StringRecordsIter<&[u8]>) -> TrainingData {
    let mut n_records: TrainingData = vec![];
    for record in records {
        let mut contains_char = false;
        let mut n_data: Vec<SetType> = vec![];
        let record = record.unwrap();
        let record = record.iter();

        for data in record {
            let n_value = match data.parse::<SetType>() {
                Ok(v) => Some(v),
                Err(_) => {
                    contains_char = true;
                    break;
                }
            }.unwrap();

            n_data.push(n_value);
        }

        if !contains_char {
            n_records.push(n_data);
        }
    }

    return n_records;
}

pub fn load_csv(file_path: &str) -> TrainingData {
    let content = read_to_string(file_path).unwrap();

    let mut reader = csv::Reader::from_reader(content.as_bytes());

    let clean_csv = clean_csv(reader.records());

    return clean_csv;
}

pub fn load_x(training_data: &TrainingData, target_index: usize) -> DVector<DVector<SetType>> {
    DVector::from_vec(training_data
            .iter()
            .map(
                |record| {
                    let mut data = vec![1.0];
                    for i in 0..record.len() {
                        if i != target_index {
                            data.push(record[i]);
                        }
                    }
                    DVector::from_vec(data)
                }
            )
            .collect()
    )
}

pub fn load_y(training_data: &TrainingData, target_index: usize) -> DVector<SetType> {
    DVector::from_vec(
    training_data
        .iter()
        .map(
            |record| {
                record[target_index]
            }
        )
        .collect()
    )
}