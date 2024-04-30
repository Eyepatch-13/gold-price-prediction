use polars::prelude::*;
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use std::error::Error;
use ndarray::Axis;

fn main() -> Result<(), Box<dyn Error>> {
    // Load data from CSV into a DataFrame
    let df = CsvReader::from_path("src/goldstock.csv")?
        .has_header(true)
        .finish()?;

    // Select relevant features
    let features = df.select(["Volume", "Open", "Low", "High"]).unwrap();

    // Extract the target variable
    let target = df.select(["Close"]).unwrap();

    // Split the data into features and target
    let features: ndarray::Array2<f64> = features.to_ndarray::<Float64Type>(IndexOrder::C)?;
    let target: ndarray::Array1<f64> = target.to_ndarray::<Float64Type>(IndexOrder::C)?.to_owned().into_shape(target.height()).unwrap();

    // Split the data into training and testing sets
    let (train_features, test_features) = features.view().split_at(Axis(0),  (0.8 * features.nrows() as f64) as usize);
    let (train_target, test_target) = target.view().split_at(Axis(0), (0.8 * features.nrows() as f64) as usize);

    let (train_features, test_features) = (train_features.to_owned(), test_features.to_owned());
    let (train_target, test_target) = (train_target.to_owned(), test_target.to_owned());

    let train_dataset = Dataset::new(train_features.clone(), train_target.clone());
    let _test_dataset = Dataset::new(test_features.clone(), test_target.clone());

    let model = LinearRegression::default().fit(&train_dataset)?; 
    let predictions = model.predict(&test_features);

    // Evaluate the model's performance using the predictions
    let mse = mean_squared_error(&test_target, &predictions);
    let rmse = mse.sqrt();
    let mae = mean_absolute_error(&test_target, &predictions);
    let r2 = r2_score(&test_target, &predictions);

    println!("Mean Squared Error: {}", mse); // 20.00765465999868
    println!("Root Mean Squared Error: {}", rmse); // 4.472991690132979
    println!("Mean Absolute Error: {}", mae); // 3.3355890412224705
    println!("R-squared (R2) Score: {}", r2); // 0.9967483572226277

    Ok(())
}

fn mean_squared_error(y_true: &ndarray::Array1<f64>, y_pred: &ndarray::Array1<f64>) -> f64 {
    ((y_true - y_pred).mapv(|x| x.powi(2)).sum()) / (y_true.len() as f64)
}

fn mean_absolute_error(y_true: &ndarray::Array1<f64>, y_pred: &ndarray::Array1<f64>) -> f64 {
    (y_true - y_pred).mapv(|x| x.abs()).sum() / (y_true.len() as f64)
}

fn r2_score(y_true: &ndarray::Array1<f64>, y_pred: &ndarray::Array1<f64>) -> f64 {
    let ss_res: f64 = (y_true - y_pred).mapv(|x| x.powi(2)).sum();
    let ss_tot: f64 = (y_true - y_true.mean().unwrap()).mapv(|x| x.powi(2)).sum();
    1.0 - (ss_res / ss_tot)
}

