use na::{DVector, dvector, DMatrix};
use rand::Rng;

pub type SetType = f64;
pub type HypoType = fn(parameters: DVector<SetType>,x: &DVector<SetType>) -> SetType;

/*
 * The first element of the x variable must be 1.0
 */
pub fn hypothesis_function(parameters: DVector<SetType>, x: &DVector<SetType>) -> SetType {
    // hø(x) = ø[0]x[0] + x[n]ø[n]...
    let value = parameters.transpose() * x;
    return value[0];
}

pub fn cost_function(predictions: &DVector<SetType>, training_result: &DVector<SetType>) -> SetType {
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

pub fn gradient_descent(
    theta: &DVector<SetType>,
    x: &DMatrix<SetType>,
    y: &DVector<SetType>,
    alpha: SetType,
) -> DVector<SetType> {
    let m = x.column(0).len();
    let mut updated_parameters = dvector![];

    for (parameter_index, parameter) in theta.iter().enumerate() {
        let predicted_values = x * theta;
        let parameters = x.column(parameter_index);
        let cost = predicted_values - y;

        let mut r: i16 = -1;
        let costs_times_params = cost
        .map(|c| {
            r += 1;
            c * parameters.row(r as usize)[0]
        });

        let sum = costs_times_params.sum();
        let slope = (1.0 / m as f64) * sum;
        let updated_param = parameter - alpha * slope;
        updated_parameters = updated_parameters.push(updated_param);

    }


    return updated_parameters;
}

pub fn normal_equation(
    x: &DMatrix<SetType>,
    training_result: &DVector<SetType>,
) -> DVector<SetType> {

    let x_transposed = x.transpose();
    
    let inverse = (x_transposed.clone() * x).pseudo_inverse(0.0).unwrap();
    
    let params = inverse * x_transposed * training_result;    

    return params;
}

pub fn prediction_cost(theta: &DVector<SetType>, x: &DMatrix<SetType>, y: &DVector<SetType>) -> SetType {
    let predictions: DVector<SetType> = x * theta;
    
    let cost = cost_function(&predictions, &y);

    return cost;
}

pub fn initialize_parameters(data_fields: u16) -> DVector<SetType> {
    let mut rng = rand::thread_rng();
    let mut parameters: DVector<SetType> = dvector![0.0];

    for _ in 0..data_fields {
        let param: SetType = rng.gen_range(0.0..1000.0);
        parameters = parameters.push(param);
    }

    return parameters;
}
