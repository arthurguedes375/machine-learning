extern crate nalgebra as na;
extern crate csv;
extern crate rand;

pub mod ml;
pub mod training_csv;

use std::io::{Write};
use std::sync::mpsc::channel;
use ctrlc;
use na::{DVector};
use training_csv::{load_csv, load_x, load_y};
use ml::{normal_equation, prediction_cost, gradient_descent, initialize_parameters};
 
fn main() {
    // Settings
    let settings = ml::Settings::load();

    // Sets up the ctrl+c handler
    let (ctx, crx) = channel();
    ctrlc::set_handler(move || ctx.send(()).expect("Could not send signal on channel."))
        .expect("Error setting Ctrl-C handler");

    // Loads the data
    let training_data = load_csv(&settings.training_data_file_path);
    let training_data_x = load_x(&training_data, settings.target_field_index);
    let training_data_y = load_y(&training_data, settings.target_field_index);

    let data_fields = training_data.row(0).len() as u16 - 1;

    // Initializes the parameters with random values
    let parameters;
    if let Some(element) = settings.default_param {
        parameters = DVector::from_element(training_data.row(0).len(), element);
    } else {
        parameters = initialize_parameters(data_fields)
    }

    // Calculates the cost of the randomly generated parameters
    let old_cost = prediction_cost(&parameters, &training_data_x, &training_data_y);
    
    // Using Gradient Descent
    let mut gradient_params = parameters.clone();
    let mut g_cost = 0.0;
    let mut iter = 0;
    loop {
        if let None = settings.learning_rate {
            println!("Not running gradient descent. Missing LEARNING_RATE");
            break;
        }
        // If it receives a ctrl + c it leaves the loop
        if let Ok(_) = crx.try_recv() {
            break;
        }

        iter += 1;

        // Apply gradient descent to the parameters
        gradient_params = gradient_descent(
            &gradient_params,
            &training_data_x,
            &training_data_y,
            settings.learning_rate.unwrap(),
        );
        
        // Calculates the cost of the adjusted parameters
        g_cost = prediction_cost(&gradient_params, &training_data_x, &training_data_y);

        // Prints out the costs
        print!("\rOld: {old_cost}, New: {g_cost}\r");
        std::io::stdout().flush().unwrap();
        
        // If the error rate is less than this then the algorithm has probably converged already   
        if g_cost < settings.minimum_error_rate || (settings.max_iters > 0 && iter >= settings.max_iters) {
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
