use crate::ml::SetType;
use std::fs::read_to_string;

use na::{DVector};
use csv::{StringRecordsIter};
use nalgebra::DMatrix;

pub type TrainingData = DMatrix<SetType>;

pub fn clean_csv(records: StringRecordsIter<&[u8]>) -> TrainingData {
    let mut n_records = vec![];
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
    let mut matrix = vec![];

    for row in &n_records {
        matrix.extend(row);
    }

    let matrix = DMatrix
        ::from_vec(n_records[0].len(), n_records.len(), matrix).transpose();

    return matrix;
}

pub fn load_csv(file_path: &str) -> TrainingData {
    let content = read_to_string(file_path).unwrap();

    let mut reader = csv::Reader::from_reader(content.as_bytes());

    let clean_csv = clean_csv(reader.records());

    return clean_csv;
}

pub fn load_x(training_data: &TrainingData, target_index: usize) -> TrainingData {    
    training_data.clone().remove_column(target_index).insert_column(0, 1.0)
}

pub fn load_y(training_data: &TrainingData, target_index: usize) -> DVector<SetType> {
    DVector::from_column_slice(training_data.column(target_index).as_slice())
}