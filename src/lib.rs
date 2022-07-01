pub mod module_isolation_forest;



#[cfg(test)]
mod tests {
    use ndarray::{Array,Array2,Axis, concatenate};
    use super::*;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::{Normal, Uniform};
    
    // Make training data
    fn make_train_data(mean:f64, std_dev:f64, dim: usize) -> Array2<f64>{
        // Multiple normal distributions
        let train1:Array2<f64> = Array::random((100, dim), Normal::new(mean * 0.3 + 2.0, std_dev * 0.3).unwrap());
        let train2:Array2<f64> = Array::random((100, dim), Normal::new(mean * 0.5 - 2.0, std_dev * 0.5).unwrap());
        let arr_train = concatenate(Axis(0), &[train1.view(), train2.view()]).unwrap();
        arr_train  
    }

    fn make_isotree_ens(mean:f64,std_dev:f64, dim:usize) -> module_isolation_forest::isolation_forest::IsolationTreeEnsemble {        
        let arr_train:Array2<f64> = make_train_data(mean, std_dev, dim);
        let mut isotreeens = module_isolation_forest::isolation_forest::IsolationTreeEnsemble::new(0, 400);
        isotreeens.fit(&arr_train);
        isotreeens
    }

    // Normal data tests
    #[test]
    fn fit_test_normal(){

        // Make normal training data
        let mean:f64 = 0.0; // mean
        let std_dev:f64 = 1.0; // standard deviation
        let dim = 5; // dimension

        let mut isotreeens = make_isotree_ens(mean, std_dev, dim);


        // Test data (normal)
        let test_normal_num = 40;
        let test_normal1:Array2<f64> = Array::random((test_normal_num/2, dim), Normal::new(mean * 0.3 + 2.0, std_dev * 0.3).unwrap());
        let test_normal2:Array2<f64> = Array::random((test_normal_num/2, dim), Normal::new(mean * 0.5 - 2.0, std_dev * 0.5).unwrap());
        let arr_test_normal:Array2<f64> = concatenate(Axis(0), &[test_normal1.view(), test_normal2.view()]).unwrap();

    
        match isotreeens.anomaly_score(&arr_test_normal) {
            Ok(s) => {
                println!("Normal Tests");
                println!("{:?}", s);
                assert_eq!(s.len(), test_normal_num);
            },
            Err(_) => {}
        }


    }

    // Anomaly data test
    #[test]
    fn fit_test_anomaly(){

        // Make normal training data
        let mean:f64 = 0.0; // mean
        let std_dev:f64 = 1.0; // standard deviation
        let dim = 5; // dimension

        let mut isotreeens = make_isotree_ens(mean, std_dev, dim);


        // Test data (anomaly)
        // Uniform distribution
        let test_anomaly_num = 40;
        let arr_test_anomaly:Array2<f64> = Array::random((test_anomaly_num, dim), Uniform::new(-4.0, 4.0));

        match isotreeens.anomaly_score(&arr_test_anomaly) {
            Ok(s) => {
                println!("Anomaly Tests");
                println!("{:?}", s);
                assert_eq!(s.len(), test_anomaly_num);
            },
            Err(_) => {}
        }

    }

    #[test]
    fn get_dim_test(){
        
        let mean:f64 = 0.0;
        let std_dev:f64 = 1.0;
        let dim = 6; 
        
        let mut isotreeens = make_isotree_ens(mean, std_dev, dim);

        assert_eq!(isotreeens.get_dim(), dim);
    }

    #[test]
    fn dim_err_test() {

        let mean:f64 = 0.0; 
        let std_dev:f64 = 1.0;
        let dim = 5; // Dimension number of training data
        
        let mut isotreeens = make_isotree_ens(mean, std_dev, dim);

        let test_dim = 4; // Dimension number of test data 

        let test_anomaly_num = 40;
        let arr_test_anomaly:Array2<f64> = Array::random((test_anomaly_num, test_dim), Uniform::new(-4.0, 4.0));

        assert_eq!(isotreeens.get_dim(), dim);
        match isotreeens.anomaly_score(&arr_test_anomaly) {
            Ok(_) => {
            },
            Err(dim_size) => {
                println!("Training data dim size:{:?}, Test data dim size:{:?}", dim, dim_size);
                assert_eq!(dim_size, test_dim);
            }
        }
    }
}