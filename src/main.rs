extern crate nalgebra as na;
extern crate csv;
extern crate rand;

pub mod ml;
pub mod training_csv;

use std::io::{Write};
use std::sync::mpsc::channel;
use ctrlc;
use training_csv::{load_csv, load_x, load_y};
use ml::{SetType, normal_equation, prediction_cost, hypothesis_function, gradient_descent, initialize_parameters};
 
fn main() {
    // Settings
    const TARGET_FIELD_INDEX: usize = 1;
    const TRAINING_DATA_FILE_PATH: &str = "./model/tst.csv";
    const LEARNING_RATE: SetType = 0.01;
    const MINIMUM_ERROR_RATE: SetType = 0.000000000000005;

    // Sets up the ctrl+c handler
    let (ctx, crx) = channel();
    ctrlc::set_handler(move || ctx.send(()).expect("Could not send signal on channel."))
        .expect("Error setting Ctrl-C handler");

    // Loads the data
    let training_data = load_csv(TRAINING_DATA_FILE_PATH);
    let training_data_x = load_x(&training_data, TARGET_FIELD_INDEX);
    let training_data_y = load_y(&training_data, TARGET_FIELD_INDEX);

    let data_fields = training_data[0].len() as u16 - 1;

    // Initializes the parameters with random values
    let parameters = initialize_parameters(data_fields);

    // Calculates the cost of the randomly generated parameters
    let old_cost = prediction_cost(&parameters, &training_data_x, &training_data_y);
    
    // Using Gradient Descent
    let mut gradient_params = parameters.clone();
    let mut g_cost = 0.0;
    loop {
        // If it receives a ctrl + c it leaves the loop
        if let Ok(_) = crx.try_recv() {
            break;
        }

        // Apply gradient descent to the parameters
        gradient_params = gradient_descent(
            &gradient_params,
            hypothesis_function,
            &training_data_x,
            &training_data_y,
            LEARNING_RATE
        );
        
        // Calculates the cost of the adjusted parameters
        g_cost = prediction_cost(&gradient_params, &training_data_x, &training_data_y);

        // Prints out the costs
        print!("\rOld: {old_cost}, New: {g_cost}\r");
        std::io::stdout().flush().unwrap();
        

        // If the error rate is less than this then the algorithm has probably converged already   
        if g_cost < MINIMUM_ERROR_RATE {
            break;
        }
    }

    // Using Normal Equation
    let ne_parameters = normal_equation(
        &training_data_x, 
        &training_data_y
    );
    let ne_cost = prediction_cost(&ne_parameters, &training_data_x, &training_data_y);


    println!("\n\nParams:\nNormal Equation: {ne_parameters}\nGradient Descent: {gradient_params}");

    

    println!("Costs:\nOld: {old_cost} \nNormal Equation: {ne_cost}\nGradient Descent: {g_cost}");
}
