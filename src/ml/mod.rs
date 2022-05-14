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

pub fn cost_function_derivative(
    parameters: &DVector<SetType>,
    hypothesis: HypoType,
    x: &DVector<DVector<SetType>>,
    training_result: &DVector<SetType>,
    parameter_index: usize,
) -> SetType {
    /*( 1 ) Sum( ( hø(x[n]) - y[n]) * x[i][parameter_index] )
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

pub fn gradient_descent(
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

pub fn normal_equation(
    x: &DVector<DVector<SetType>>,
    training_result: &DVector<SetType>,
) -> DVector<SetType> {

    let mut matrix = vec![];

    for row in x {
        matrix.extend(row);
    }

    let x = DMatrix::from_vec(x[0].len(), x.len(), matrix).transpose();

    let x_transposed = x.transpose();
    
    let inverse = (x_transposed.clone() * x).pseudo_inverse(0.1).unwrap();
    
    let params = inverse * x_transposed * training_result;    

    return params;
}

pub fn prediction_cost(parameters: &DVector<SetType>, x: &DVector<DVector<SetType>>, y: &DVector<SetType>) -> SetType {
    let mut predictions: DVector<SetType> = dvector![];

    for x in x.iter() {
        let prediction = hypothesis_function(parameters.clone(), &x);
        predictions = predictions.push(prediction);
    }
    
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
