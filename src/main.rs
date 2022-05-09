extern crate nalgebra as na;

use std::io::{self, Write};
use na::{DVector, dvector};

type SetType = f64;
type HypoType = fn(parameters: DVector<SetType>,x: &DVector<SetType>) -> SetType;


/*
 * The first element of the x variable must be 1.0
 */
fn hypothesis_function (parameters: DVector<SetType>, x: &DVector<SetType>) -> SetType {
    // hø(x) = ø[0]x[0] + x[n]ø[n]...
    let value = parameters.transpose() * x;
    return value[0];
}

fn cost_function (predictions: &DVector<SetType>, training_result: &DVector<SetType>) -> SetType {
    /*( 1  ) Sum( ( hø(x[n]) - y[n]) ** 2 )
      ( 2m ) 
    */
    let mut sum = 0.0;
    let m = training_result.len() - 1;
    for i in 0..m {
        sum += (predictions[i] - training_result[i]).powf(2.0);
    }

    let cost = (1.0 / (2.0 * m as SetType)) * sum;

    return cost;
}

fn cost_function_derivative (
    parameters: &DVector<SetType>,
    hypothesis: HypoType,
    x: &DVector<DVector<SetType>>,
    training_result: &DVector<SetType>,
    parameter_index: usize,
) -> SetType {
    /*( 1  ) Sum( ( hø(x[n]) - y[n]) * x[i][parameter_index] )
      ( m ) 
    */
    let mut sum = 0.0;
    let m = training_result.len() - 1;
    for i in 0..m {
        let y = training_result[i];
        sum += (hypothesis(parameters.clone(), &x[i]) - y) * x[i][parameter_index];
    }

    let cost = (1.0 /  m as SetType) * sum;

    return cost;
}

fn gradient_descent(
    parameters: &DVector<SetType>,
    hypothesis: HypoType,
    x: &DVector<DVector<SetType>>,
    training_result: &DVector<SetType>,
    learning_rate: SetType,
) -> DVector<SetType> {
    let mut updated_parameters = dvector![];

    for (parameter_index, parameter) in parameters.iter().enumerate() {
        let slope = cost_function_derivative(
            &parameters,
            hypothesis,
            &x,
            &training_result,
            parameter_index,
        );
        let updated_param = parameter - learning_rate * slope;
        updated_parameters = updated_parameters.push(updated_param);
    }


    return updated_parameters;
}

fn main() {
    // let mut PARAMETERS: DVector<f32> = dvector![0.0, 0.5];
    let mut parameters: DVector<SetType> = dvector![10.0, -2.0];

    let training_data_x: DVector<DVector<SetType>> = dvector![
        dvector![1.0, 1.0],
        dvector![1.0, 2.0],
        dvector![1.0, 4.0]
    ];
    let training_data_y: DVector<SetType> = dvector![
        0.5,
        1.0,
        2.0
    ];

    let mut predictions: DVector<SetType> = dvector![];

    for x in training_data_x.iter() {
        let prediction = hypothesis_function(parameters.clone(), &x);
        predictions = predictions.push(prediction);
    }

    
    let old_cost = cost_function(&predictions, &training_data_y);
    
    loop {
        parameters = gradient_descent(
            &parameters,
            hypothesis_function,
            &training_data_x,
            &training_data_y,
            0.5
        );
        let mut n_predictions: DVector<SetType> = dvector![];

        for x in training_data_x.iter() {
            let prediction = hypothesis_function(parameters.clone(), &x);
            n_predictions = n_predictions.push(prediction);
        }
        
        let n_cost = cost_function(&n_predictions, &training_data_y);
        print!("\rOld: {old_cost}, New: {n_cost}\r");
        std::io::stdout().flush().unwrap();

        if n_cost < 0.000005 {
            break;
        }
    }

    println!("{parameters}");
}
