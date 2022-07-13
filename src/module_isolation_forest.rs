pub mod isolation_forest {
    extern crate ndarray;
    use ndarray::{Array,ArrayView1,Array2,ShapeError};

    extern crate rand;
    use rand::{thread_rng, Rng};
    use rand::distributions::{Uniform};
    use rand_distr::{Distribution};
    
    extern crate statrs;

    use std::thread;
    use std::sync::{Arc, Mutex};
    
    // use std::panic;
    
    extern crate rayon;
    use rayon::prelude::*;
    
    use serde::{Serialize, Deserialize};
    // Node data setting
    // It's defined the nodes as enums for the recursive processing,
    // binary tree elements in the middle and leaf elements at the end.
    #[derive(Serialize, Deserialize, Clone)]
    pub enum IsolationNode {
        // Branches information on binary tree
        Decision {
            left : Box<IsolationNode>, // Nodes smaller than the boundary
            right: Box<IsolationNode>, // Nodes above the boundary
            split_att: usize, // Column number of the variable selected in the split
            split_val: f64, // Boundary value
        },
        // Leaf information 
        // Data size(rows number) is recorded.
        // (Not include data.)
        Leaf {
            size: usize,
        }
    }

    // Node
    impl IsolationNode {
        // For branches
        fn new_decision(
            left:Box<IsolationNode>, 
            right:Box<IsolationNode>, 
            split_att:usize,
            split_val: f64
            ) -> Box<IsolationNode>{
            let node = IsolationNode::Decision {
                left: left,
                right: right,
                split_att: split_att,
                split_val: split_val,
            };
            Box::new(node)
        }

        // For leaf
        fn new_leaf(
            size:usize, 
            //data:Option<Array2<f64>>
            ) -> Box<IsolationNode>{

            let node = IsolationNode::Leaf {
                size: size, // data number
            };
            Box::new(node)
        }
    }


    // One tree of Isolation forest 
    #[derive(Serialize, Deserialize)]
    pub struct IsolationTree {
        height: u32, 
        height_limit: u32, 
    }

    impl IsolationTree {

        fn new(height:u32, height_limit:u32) -> Self{
            IsolationTree {height : height, height_limit : height_limit}
        }

        // Pass the training data(Array2<f64>) and fit.
        fn fit(& mut self, x:&Array2<f64>) -> Result< Box<IsolationNode>, ShapeError>{
            
            // When data is isolated or at the specified depth, return the leaf.
            if x.nrows() <= 1 || self.height >= self.height_limit {
                let node = IsolationNode::new_leaf(x.nrows());            
                return Ok(node);
            }
            
            // Extract columns randomly.
            let mut rng = thread_rng();
            let split_att: usize = rng.gen_range(0..x.ncols());
            
            // Select the column.
            let col = x.column(split_att);
            let vec_col = col.t().to_vec();

            // Select the max value and the min value.
            let max = vec_col.iter().fold(0.0/0.0, |m, &v: &f64| v.max(m));
            let min = vec_col.iter().fold(0.0/0.0, |m, &v: &f64| v.min(m));
    
            // Determine the thresholds.
            let mut rng = thread_rng();
            let split_val:f64 = if min == max {
                min
            }
            else {
                rng.gen_range(min..max)
            };
        
            // Split branches.       
            let mut x_left:Array2<f64> = Array::zeros((0,x.ncols())); //　Initialize with row number = 0
            let mut x_right:Array2<f64> = Array::zeros((0,x.ncols())); // Initialize with row number = 0
            
    
            for i in x.rows(){
                if i.to_vec()[split_att] < split_val {
                    // Add row on th left.
                    x_left.push_row(i)?;
                }
                else {
                    // Add row on the right.
                    x_right.push_row(i)?;
                }
            }
            let height = self.height;
            let height_limit = self.height_limit;

            // Create a branch, add 1 depth, set the branching condition, and return the branch.
            let left_node = IsolationTree::new(height + 1, height_limit).fit(&x_left)?;        
            let right_node = IsolationTree::new(height + 1, height_limit).fit(&x_right)?;

            let node = IsolationNode::new_decision(left_node, right_node, split_att, split_val);
            Ok(node)
        }
    }

    // Tree set for ensemble learning
    #[derive(Serialize, Deserialize)]
    pub struct IsolationTreeEnsemble {
        sample_size: usize,
        n_trees: usize,
        tree_set: Vec<IsolationNode>,
        dim: usize, // dimension number
    }
    
    impl IsolationTreeEnsemble {
        pub fn new(sample_size:usize, n_trees:usize) -> Self{
            IsolationTreeEnsemble {
                sample_size: sample_size, 
                n_trees:n_trees, 
                tree_set: Vec::new(),
                dim : 0,         
            }
        }

        fn c(sample_size:usize) -> f64{
            const EULER_GAMMA:f64 = 0.5772156649; // Euler-Mascheroni constant
        
            if sample_size > 1 {
                let sizef:f64 = sample_size as f64;
                
                return  (2. * ((sizef - 1.).ln() + EULER_GAMMA) ) - ((2. *(sizef -1.))/sizef);        
            }
            else { // sample_size = 1
                return 0.;
            }
        }

        // Pass the data and return length.
        fn tree_path_length(node:Box<&IsolationNode>, x:&ArrayView1<f64>)->(usize, usize){
            match *node {
                IsolationNode::Decision {left, right, split_att, split_val}=> {
                    let direction = if x.to_vec()[*split_att] < *split_val {
                        left
                    }
                    else {
                        right
                    };

                    let result = Self::tree_path_length(Box::new(direction), x);
                    let length = result.0 + 1;
                    let decision_size = result.1;
            
                    return (length, decision_size);
                }

                IsolationNode::Leaf {size} => {
                    let length = 1;
                    let size = *size;
                    return (length, size);
                }
            }
        }

        fn make_isotree(x:&Array2<f64>, sample_size:usize, height_limit:u32) -> Result<IsolationNode, ShapeError>{        
            let mut rng = thread_rng();
            let data_range = Uniform::new_inclusive(0, x.nrows() - 1);
            let data_rows: Vec<usize> = data_range.sample_iter(&mut rng).take(sample_size).collect();
            let mut random_data:Array2<f64> = Array::zeros((0,x.ncols()));

            for i in data_rows.iter(){
                random_data.push_row(x.row(*i))?;
            }

            // Make one isolation tree.
            let mut isotree = IsolationTree::new(0, height_limit);
            let data_node = isotree.fit(&random_data)?;
            
            Ok(*data_node)
        }


        // Training
        pub fn fit(&mut self, x:&Array2<f64>) {
            self.dim = x.ncols(); // dimension number = columns number

            if self.sample_size == 0{
                self.sample_size = x.nrows();
            }

            let height_limit:u32 = (self.sample_size as f64).log2().ceil() as u32;
            let sample_size = self.sample_size;
            
            let x_tmp = Arc::new(x.clone()); // The training data is for reference only.
            let tree_set = Arc::new(Mutex::new(vec!{}));

            for _ in 0..self.n_trees {
                let x_t = Arc::clone(&x_tmp);
                let t_set = Arc::clone(&tree_set);

                let handle = thread::spawn(move || {
                    let data = Self::make_isotree(&x_t, sample_size, height_limit);
                    match data {
                        // Compose a tree with only error-free items
                        Ok(ret) => {t_set.lock().unwrap().push(ret);},
                        Err(_) => {},
                    }
                });
                handle.join().unwrap();
            }
            self.tree_set =  tree_set.lock().unwrap().to_vec();
            
            /*
            // Thread on rayon
            self.tree_set = (0..self.n_trees)
                .into_par_iter()
                .map(move |_|
                {
                    let data = Self::make_isotree(&x, sample_size, height_limit);
                    match data {
                        Ok(ret) => {ret},
                        Err(e) => {
                            panic!("ShapeError: {:?}",e);
                        },
                    }
                })
                .collect();
            */
        }

        // Get length mean
        fn get_path_length_mean(&self, row:&ArrayView1<f64>)-> f64{
            // thread on rayon
            let path:Vec<f64> = self.tree_set
                .par_iter()
                .map(|tree| {
                    let result = Self::tree_path_length(Box::new(tree), row);
                    // If default depth is reached before isolation, it is adjusted.
                    (result.0 as f64) + Self::c(result.1)            
                })
                .collect();

            // mean per data
            let mut sum: f64 = 0.0;
            for j in path.iter() {
                sum += *j as f64;
            }
            let path_mean:f64 = sum/(path.len() as f64);
            path_mean
        }



        // Length mean
        fn path_length_mean(&self, x:&Array2<f64>) -> Vec<f64>{
            let x_rows: Vec<_> = x.outer_iter().collect();
            let paths_mean: Vec<f64> = x_rows
                .par_windows(1) 
                .map(|w| self.get_path_length_mean(&w[0]))
                .collect();
            paths_mean
        }

        // Get dimension number.
        pub fn get_dim(&mut self) -> usize {
            self.dim
        }

        // Get anomaly scores.
        pub fn anomaly_score(&mut self, x:&Array2<f64>)  -> Result<Vec<f64>, usize> {

            // Whether the number of dimensions matches。
            if self.dim != x.ncols() {
                // If unmatched, return error.
                return Err(x.ncols());
            }

            // Get the length mean per data, and get the anomaly scores.
            let sample_size = self.sample_size;    
            let paths_mean = self.path_length_mean(x);

            let anomaly_scores:Vec<f64> = paths_mean
                .par_iter()
                .map(|l| (2. as f64).powf((-1.* l)/(Self::c(sample_size) as f64)))
                .collect();
            Ok(anomaly_scores)
        }

    }
}


