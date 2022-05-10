extern crate nalgebra as na;
extern crate csv;
extern crate rand;

pub mod ml;
pub mod training_csv;

use std::io::{Write};
use std::sync::mpsc::channel;
use ctrlc;
use training_csv::{load_csv, load_x, load_y};
use ml::{SetType, prediction_cost, hypothesis_function, gradient_descent, initialize_parameters};

fn main() {
    // Settings
    const LEARNING_RATE: SetType = 0.00000000865;
    const DATA_FIELDS: u16 = 34 - 22;
    const TRAINING_DATA_FILE_PATH: &str = "./model/short_training_data.csv";
    const MINIMUM_ERROR_RATE: SetType = 0.000005;

    // Sets up the ctrl+c handler
    let (ctx, crx) = channel();
    ctrlc::set_handler(move || ctx.send(()).expect("Could not send signal on channel."))
        .expect("Error setting Ctrl-C handler");

    // Loads the data
    let training_data = load_csv(TRAINING_DATA_FILE_PATH);
    let training_data_x = load_x(&training_data);
    let training_data_y = load_y(&training_data);

    // Initializes the parameters with random values
    let mut parameters = initialize_parameters(DATA_FIELDS);

    // Calculates the cost of the randomly generated parameters
    let old_cost = prediction_cost(&parameters, &training_data_x, &training_data_y);
    
    loop {
        // If it receives a ctrl + c it leaves the loop
        if let Ok(_) = crx.try_recv() {
            break;
        }

        // Apply gradient descent to the parameters
        parameters = gradient_descent(
            &parameters,
            hypothesis_function,
            &training_data_x,
            &training_data_y,
            LEARNING_RATE
        );
        
        // Calculates the cost of the adjusted parameters
        let n_cost = prediction_cost(&parameters, &training_data_x, &training_data_y);

        // Prints out the costs
        print!("\rOld: {old_cost}, New: {n_cost}\r");
        std::io::stdout().flush().unwrap();

        // If the error rate is less than this then the algorithm has probably converged already   
        if n_cost < MINIMUM_ERROR_RATE {
            break;
        }
    }

    println!("{parameters}");
}
