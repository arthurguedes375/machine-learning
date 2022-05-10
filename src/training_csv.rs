use crate::ml::SetType;
use std::fs::read_to_string;

use na::{DVector};

pub type TrainingData = Vec<Vec<SetType>>;

pub fn load_csv(file_path: &str) -> TrainingData {
    let content = read_to_string(file_path).unwrap();

    let mut reader = csv::Reader::from_reader(content.as_bytes());

    return reader
    .records()
    .map(|record|{
        record
            .unwrap()
            .iter()
            .map(|data| {
                data.parse::<SetType>().unwrap()
            })
            .collect::<Vec<SetType>>()
    })
    .collect();
}

pub fn load_x(training_data: &TrainingData) -> DVector<DVector<SetType>> {
    DVector::from_vec(
        training_data
            .iter()
            .map(
                |record| {
                    let mut data = vec![1.0];
                    data.extend(record[..record.len() - 1].to_vec());
                    DVector::from_vec(data)
                }
            )
            .collect()
    )
}

pub fn load_y(training_data: &TrainingData) -> DVector<SetType> {
    DVector::from_vec(
    training_data
        .iter()
        .map(
            |record| {
                record[record.len() - 1]
            }
        )
        .collect()
    )
}