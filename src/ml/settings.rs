use std::{env, str::FromStr, fmt};

pub struct Settings {
    pub target_field_index: usize,
    pub training_data_file_path: String,
    pub learning_rate: Option<f64>,
    pub minimum_error_rate: f64,
    pub max_iters: usize,
    pub default_param: Option<f64>,
}

impl Settings {
    pub fn env<T>(env_var_name: &str) -> Option<T>
        where T: FromStr,
            <T as FromStr>::Err: fmt::Debug,  
    {
        match env::var(env_var_name) {
            Ok(v) => Some(v.parse::<T>().unwrap()),
            Err(_) => None,
        }
    }
    
    pub fn env_or<T>(env_var_name: &str, default: T) -> T
        where T: FromStr,
            <T as FromStr>::Err: fmt::Debug,
    {
        match Settings::env(env_var_name) {
            Some(v) => v,
            None => default,
        }
    }

    pub fn load() -> Settings {
        Settings {
            target_field_index: Settings
                ::env_or("TARGET_FIELD_INDEX", 1),

            training_data_file_path: Settings
                ::env_or("DATA_FILE_PATH", "model/training.csv".into()),

            learning_rate: Settings
                ::env("LEARNING_RATE"),

            minimum_error_rate: Settings
                ::env_or("MIN_ERR_RATE", 0.000000000000005),

            max_iters: Settings
                ::env_or("MAX_ITERS", 0),
            
            default_param: Settings
                ::env("DEFAULT_PARAM"),
        }
    }

}