'''
This code is a copy of the paper:
    Title       : "Evaluating robustness of tabular models under meta-features based shifts",
    Paper Link  : https://openreview.net/pdf?id=BS68QcFppq
    GitHub repo : https://github.com/ITMO-NSS-team/OOD_Tab_Evaluation.git

Modifications:
    generation: 25 (initially 300)
    population: 10 (initially 50)
    Take the top solution
    test_size: 0.3 (initially 0.2)
    main() accepts pd.DataFrame instead of re-read data, remove file_name
    use_imbalance -> omit imbalance used in the original implementation, unsuitable for regression target
    Optimize _evaluate_split() using boolean mask in lieu of trivial list comprehension
    Add new variable "idx" to keep track the indices of current seed for random_state
    import multiprocessing -> modify _setup_deap() and add __del__()
    save_pareto_solution() -> remove train/test pair information summary and save as .parquet files
    create_convergence_plot() -> unecessary
    remove samples in extensive dataset -> keep 800,000 samples for the sake of speed and efficiency
'''
import numpy as np
import random
from deap import base, creator, tools, algorithms

from pymfe.mfe import MFE
import pandas as pd
from collections import Counter
import warnings

import matplotlib.pyplot as plt

import os

warnings.filterwarnings('ignore')


def extract_metafeatures(X, y=None, meta_features=None):
    """Extracts metafeatures from data X and y"""
    if meta_features is None:
        meta_features = ["class_conc"]
    
    mfe = MFE(features=meta_features, summary="mean")
    if y is not None:
        mfe.fit(X, y)       
    else:
        mfe.fit(X)
    
    ft_names, ft_values = mfe.extract()
    
    return dict(zip(ft_names, ft_values))
def distances_per_metafeature(meta1, meta2):
    """Simple directed distance - encourage train >= test"""
    distances = {}
    
    for feature in meta1:
        if feature in meta2:
            val1 = meta1[feature]
            val2 = meta2[feature]
            
            if val1 != 0 and val2 != 0:
                        ratio = val1 / val2
                        distances[feature] = ratio

    
    return distances


def class_imbalance_ratio(y):
    """Computes class imbalance ratio"""
    class_counts = Counter(y)
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    
    return min_count / max_count


class EvolutionarySplitOptimizer:
    """Class for evolutionary optimization of data splitting"""
    
    def __init__(self, X, y, test_size=0.3, meta_features=None, use_imbalance = False,
                 population_size=100, generations=30, random_state=None, file=None, idx = None):
        """Initialize optimizer"""
        # Convert to numpy arrays for correct indexing
        self.X = np.array(X)
        self.y = np.array(y)
        self.test_size = test_size
        # Use all specified metafeatures
        self.meta_features = meta_features if meta_features else ["attr_ent", "class_conc", "max", "min",  "range", "sparsity"]
        self.use_imbalance = use_imbalance
        self.population_size = population_size
        self.generations = generations
        self.n_samples = self.X.shape[0]
        self.test_indices_size = int(self.n_samples * test_size)
        self.file = file

        if idx is None:
            raise Exception("Idx is not defined")
        self.idx = idx
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        # Determine number of optimization objectives
        # Create sample split to find number of metafeatures
        sample_train_indices = list(range(self.test_indices_size))
        sample_test_indices = list(range(self.test_indices_size, self.test_indices_size * 2))
        
        X_sample_train = self.X[sample_train_indices]
        y_sample_train = self.y[sample_train_indices]
        X_sample_test = self.X[sample_test_indices]
        y_sample_test = self.y[sample_test_indices]
        
        train_meta = extract_metafeatures(X_sample_train, y_sample_train, meta_features=self.meta_features)
        test_meta = extract_metafeatures(X_sample_test, y_sample_test, meta_features=self.meta_features)
        
        distances = distances_per_metafeature(train_meta, test_meta)
        self.metafeature_names = list(distances.keys())
        
        # Number of objectives = number of metafeatures + class imbalance
        self.n_objectives = len(self.metafeature_names) + (1 if use_imbalance else 0)
        # print(f"Metafeatures: {self.metafeature_names}")
        
        # Initialize hall of fame
        self.hall_of_fame = tools.HallOfFame(5)  # General hall of fame size 5
        

        
        # Setup DEAP
        self._setup_deap()

    def create_individual_by_ordering(self):
        """Creates individual based on feature ordering patterns"""
        n, d = self.X.shape
        j = np.random.randint(d)
        sorted_indices = np.argsort(self.X[:, j])
        pattern = random.choice(['first_k', 'last_k', 'contiguous', 'every_other', 'random'])
        k = self.test_indices_size
        
        if pattern == 'first_k':
            test_indices = sorted_indices[:k]
        elif pattern == 'last_k':
            test_indices = sorted_indices[-k:]
        elif pattern == 'contiguous':
            start = random.randint(0, n - k)
            test_indices = sorted_indices[start:start + k]
        elif pattern == 'every_other': 
            step = 2
            start = random.randint(0, 1)
            candidate = sorted_indices[start::step]
            test_indices = candidate[:k] if len(candidate) >= k else sorted_indices[:k]
        else:
            test_indices = random.sample(list(sorted_indices), k)
        
        return creator.Individual(list(test_indices))


    def _setup_deap(self):
        """Setup DEAP for multi-objective optimization"""
        # Clean existing types
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        # Create types for multi-objective optimization
        # weights - all objectives are maximized (1.0 for each)
        weights = [1.0] * (self.n_objectives - (1 if self.use_imbalance else 0))
        
        if self.use_imbalance:
            weights.append(-1.0)
        
        creator.create("FitnessMulti", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        self.toolbox = base.Toolbox()
        
        # Register individual creation functions with new method
        self.toolbox.register("individual", self.create_individual_by_ordering)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register evolutionary algorithm functions
        self.toolbox.register("evaluate", self._evaluate_split)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutation, indpb=0.2)
        
        self.toolbox.register("select", tools.selNSGA2)

        self._pool = None
        try:
            from multiprocessing.pool import ThreadPool
            n_workers = max(1, (os.cpu_count() or 4))
            self._pool = ThreadPool(processes=n_workers)
            self.toolbox.register("map", self._pool.map)
        except Exception:
            # Fallback to sequential map
            pass


    def _evaluate_split(self, individual):
        """Function to evaluate data split quality"""
        # Get sample indices
        # test_indices = individual
        # train_indices = [i for i in range(self.n_samples) if i not in test_indices]

        test_indices = np.asarray(individual)

        # Assume select all indices
        mask = np.ones(self.n_samples, dtype = bool)
        # Change to False if they are included in test_indices = not in test_indices
        mask[test_indices] = False 
        
        # Form samples - use numpy arrays
        # X_train, y_train = self.X[train_indices], self.y[train_indices]
        # X_test, y_test = self.X[test_indices], self.y[test_indices]

        X_train, y_train = self.X[mask], self.y[mask]
        X_test, y_test = self.X[~mask], self.y[~mask]
        
        # Extract metafeatures
        train_meta = extract_metafeatures(X_train, y_train, meta_features=self.meta_features)
        test_meta = extract_metafeatures(X_test, y_test, meta_features=self.meta_features)
        #print('train_meta', train_meta)
        
        # Compute distances for each metafeature
        distances = distances_per_metafeature(train_meta, test_meta)
        
        # Form objectives list in same order as self.metafeature_names
        objectives = []
        for feature_name in self.metafeature_names:
            if feature_name in distances:
                objectives.append(distances[feature_name])
            else:
                objectives.append(0.0)  # If metafeature not found
        
        # Add class imbalance as last objective
        if self.use_imbalance:
            train_imbalance = class_imbalance_ratio(y_train)
            test_imbalance = class_imbalance_ratio(y_test)
    
            imbalance_ratio = (abs(train_imbalance - test_imbalance) - 0.1*(min(train_imbalance, 1-train_imbalance) - 0.1*min(test_imbalance, 1-test_imbalance)))
    
            objectives.append(imbalance_ratio)
        
        return tuple(objectives)
    
    def _crossover(self, ind1, ind2):
        """Crossover operator with index exchange"""
        # Create parent copies
        child1 = ind1.copy()
        child2 = ind2.copy()
        
        # Convert to sets for easier work
        set1 = set(child1)
        set2 = set(child2)
        
        # Find indices that exist only in first individual
        only_in_1 = set1 - set2
        # Find indices that exist only in second individual  
        only_in_2 = set2 - set1
        
        # If there are different indices, perform exchange
        if only_in_1 and only_in_2:
            # Determine number of indices to exchange (from 1 to minimum available)
            max_exchange = min(len(only_in_1), len(only_in_2))
            n_exchange = random.randint(1, min(max_exchange, max(1, len(child1) // 4)))
            
            # Choose random indices for exchange
            indices_from_1 = random.sample(list(only_in_1), min(n_exchange, len(only_in_1)))
            indices_from_2 = random.sample(list(only_in_2), min(n_exchange, len(only_in_2)))
            
            # Perform exchange: replace indices in child1 with indices from child2
            for i, idx_from_1 in enumerate(indices_from_1):
                if i < len(indices_from_2):
                    idx_from_2 = indices_from_2[i]
                    
                    # Find positions for replacement
                    pos_in_child1 = child1.index(idx_from_1)
                    pos_in_child2 = child2.index(idx_from_2)
                    
                    # Perform exchange
                    child1[pos_in_child1] = idx_from_2
                    child2[pos_in_child2] = idx_from_1
        
        return creator.Individual(child1), creator.Individual(child2)
    
    def _mutation(self, individual, indpb=0.7):
        """Mutation operator with index replacement"""
        # Available indices for replacement
        # available_indices = [i for i in range(self.n_samples) if i not in individual]

        test_indices = np.asarray(individual)

        # Assume select all indices
        mask = np.ones(self.n_samples, dtype = bool)
        # Change to False if they are included in test_indices = not in test_indices/individual
        mask[test_indices] = False 

        available_indices = np.arange(self.n_samples)
        available_indices = available_indices[mask]

        
        if available_indices.size == 0 or random.random() > indpb:
            return individual,
        
        # Create copy of individual
        new_individual = individual.copy()
        
        # Determine number of mutations
        n_mutations = random.randint(1, min(3, len(new_individual)))
        
        # Perform mutations
        positions = random.sample(range(len(new_individual)), n_mutations)
        new_indices = random.sample(available_indices.tolist(), n_mutations)
        
        for pos, new_idx in zip(positions, new_indices):
            new_individual[pos] = new_idx
        
        return creator.Individual(new_individual),
    

    def optimize(self, verbose=True):
        """Run optimization"""

        try:
            # Create initial population
            population = self.toolbox.population(n=self.population_size)
            
            # Evaluate initial population
            for ind in population:
                ind.fitness.values = self.toolbox.evaluate(ind)
            
            # Setup statistics
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)
            
            # Pareto front for storing best solutions
            pareto_front = tools.ParetoFront()
            
            # Custom algorithm with specialized hall of fame updates
            logbook = tools.Logbook()
            logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
            
            # Evaluate initial population
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            if self.hall_of_fame is not None:
                self.hall_of_fame.update(population)
            
            record = stats.compile(population) if stats else {}
            logbook.record(gen=0, nevals=len(invalid_ind), **record)
            
            # if verbose:
            #     print(logbook.stream)
            
            # Evolutionary process
            for gen in range(1, self.generations + 1):
                # Selection and offspring creation
                offspring = algorithms.varAnd(population, self.toolbox, cxpb=0.7, mutpb=0.2)
                
                # Evaluate offspring
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
                # Select next generation
                population = self.toolbox.select(population + offspring, self.population_size)
                
                # Update general hall of fame
                if self.hall_of_fame is not None:
                    self.hall_of_fame.update(population)
                
                # Statistics
                record = stats.compile(population) if stats else {}
                logbook.record(gen=gen, nevals=len(invalid_ind), **record)
                
                # if verbose:
                #     print(logbook.stream)
            
            # Fill Pareto front with final population for multi-objective optimization
            pareto_front.update(population)
            
            # Sort solutions by first objective (Euclidean distance)
            best_solutions = sorted(self.hall_of_fame, key=lambda x: x.fitness.values[0], reverse=True)
                  
            return {
                'best_solutions': best_solutions,
                'pareto_front': pareto_front,
                'hall_of_fame': self.hall_of_fame,
                'final_population': population,
                'logbook': logbook
            }
        finally:
            if getattr(self, "_pool", None) is not None:
                try:
                    self._pool.close()
                    self._pool.join()
                except Exception:
                    pass
    
    
    def get_all_unique_solutions(self):
        """Gets all unique solutions from hall of fame"""
        all_solutions = []
        seen_solutions = set()
        
        # Add solutions from general hall of fame
        for solution in self.hall_of_fame:
            solution_tuple = tuple(sorted(solution))
            if solution_tuple not in seen_solutions:
                all_solutions.append(solution)
                seen_solutions.add(solution_tuple)
        
        return all_solutions

        
    def get_split_details(self, individual):
        """Get detailed information about split"""
        test_indices = np.asarray(individual)

        # Assume select all indices
        mask = np.ones(self.n_samples, dtype = bool)
        # Change to False if they are included in test_indices = not in test_indices
        mask[test_indices] = False 

        train_indices = np.arange(self.n_samples)
        train_indices = train_indices[mask]
        
        # Form samples - use numpy arrays
        X_train, y_train = self.X[mask], self.y[mask]
        X_test, y_test = self.X[~mask], self.y[~mask]
        
        train_meta = extract_metafeatures(X_train, y_train, meta_features=self.meta_features)
        test_meta = extract_metafeatures(X_test, y_test, meta_features=self.meta_features)
        
        # Вычисляем расстояния для каждой метахарактеристики
        distances = distances_per_metafeature(train_meta, test_meta)

        if self.use_imbalance:
            train_imbalance = class_imbalance_ratio(y_train)
            test_imbalance = class_imbalance_ratio(y_test)
        else:
            train_imbalance = None
            test_imbalance = None
        
        return {
            'train_indices': train_indices,
            'test_indices': test_indices,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_metafeatures': train_meta,
            'test_metafeatures': test_meta,
            'metafeature_distances': distances,  # Расстояния для каждой метахарактеристики
            'train_imbalance_ratio': train_imbalance,
            'test_imbalance_ratio': test_imbalance,
            'train_class_distribution': Counter(y_train),
            'test_class_distribution': Counter(y_test)
        }
    
   
    
    def save_pareto_solutions(self, pareto_front, output_dir, prefix="solution"):
        """Saves all solutions from Pareto front to parquet files"""
        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        saved_files = []
        solutions_info = []
        
        # Sort solutions by Euclidean distance (descending)
        sorted_solutions = sorted(pareto_front, key=lambda x: x.fitness.values[0], reverse=True)

        # Get the 1st solution in sorted_solution
        solution = sorted_solutions[0]
        
        # Get detailed information about solution
        details = self.get_split_details(solution)
        
        # Form file names
        train_filename = f"train_{self.idx+1}.parquet"
        test_filename = f"test_{self.idx+1}.parquet"
        # info_filename = f"{prefix}_{i+1:02d}_info.txt"
        
        train_path = os.path.join(output_dir, train_filename)
        test_path = os.path.join(output_dir, test_filename)
        # info_path = os.path.join(output_dir, info_filename)
        source_df = self.file.copy()
        if source_df.columns[0] == 'Unnamed: 0' or source_df.columns[0] =='':
            source_df = source_df.iloc[:, 1:]
        
        
        if isinstance(source_df, pd.DataFrame):
            # If source data is DataFrame, save column names
            X_train_df = source_df.iloc[details['train_indices']].copy()
            # the last column is already target, we need just rename it
            X_train_df.rename(columns={X_train_df.columns[-1]: 'target'}, inplace=True)
            
            X_test_df = source_df.iloc[details['test_indices']].copy()
            X_test_df.rename(columns={X_test_df.columns[-1]: 'target'}, inplace=True)
        else:
            # If numpy arrays, create DataFrame
            feature_names = [f'feature_{j}' for j in range(self.X.shape[1])]
            
            X_train_df = pd.DataFrame(details['X_train'], columns=feature_names)
            X_train_df['target'] = details['y_train']
            
            X_test_df = pd.DataFrame(details['X_test'], columns=feature_names)
            X_test_df['target'] = details['y_test']
            
        # Save parquet files
        X_train_df.to_parquet(train_path, index=False)
        X_test_df.to_parquet(test_path, index=False)
            
        # Add information about saved files
        solution_info = {
            'SEED_index': self.idx,
            'train_file': train_path,
            'test_file': test_path,
        }
        
        # Add distances for each metafeature
        for name, dist in details['metafeature_distances'].items():
            solution_info[f'distance_{name}'] = dist
        
        solutions_info.append(solution_info)
        saved_files.extend([train_path, test_path])
        
        # Form string with distances for each metafeature
        distances_str = ", ".join([f"{name}={dist:.4f}" for name, dist in details['metafeature_distances'].items()])
        
        print(f"Solution {self.idx+1}: distances=[{distances_str}]")
        
        # Create summary file with information about all solutions
        summary_path = os.path.join(output_dir, f"pareto_solutions_summary_{self.idx + 1}.csv")
        
        # Form data for DataFrame
        summary_data = []
        for info in solutions_info:
            row = {
                'SEED_index': info['SEED_index'],
                'Train_File': os.path.basename(info['train_file']),
                'Test_File': os.path.basename(info['test_file']),
            }
            
            # Add distances for each metafeature
            for key, value in info.items():
                if key.startswith('distance_'):
                    row[key.replace('distance_', 'Distance_').title()] = value
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_path, index=False)
        saved_files.append(summary_path)
        
        # print(f"\n📋 Created summary file: {summary_path}")
        # print(f"💾 Total files saved: {len(saved_files)}")
        # print(f"📁 Directory: {os.path.abspath(output_dir)}")
        
        return {
            'output_directory': os.path.abspath(output_dir),
            'solutions_info': solutions_info,
            'saved_files': saved_files,
            'summary_file': summary_path,
            'total_solutions': len(sorted_solutions)
        }

  
def create_convergence_plot(logbook, metafeature_names, file_prefix_name):
    """Creates convergence plot of fitness functions by generations"""
    if not logbook or len(logbook) == 0:
        print("⚠️ No data for convergence plot")
        return
    
    # Get data from logbook
    generations = [record['gen'] for record in logbook]
    
    # Create subplots for each metafeature
    n_metafeatures = len(metafeature_names)
    # n_imbalance = 1  # Class imbalance
    # n_total = n_metafeatures + 1
    n_total = n_metafeatures
    
    # Determine subplot grid size
    cols = min(3, n_total)
    rows = (n_total + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    
    # Properly handle different cases
    if n_total == 1:
        axes = np.array([axes])
    elif rows == 1:
        axes = axes.reshape(1, -1).flatten()
    else:
        axes = axes.flatten()
    
    # Plots for metafeatures
    for i, metafeature_name in enumerate(metafeature_names):
        ax = axes[i]
        
        # Extract statistics for given metafeature
        max_values = [record['max'][i] for record in logbook]
        avg_values = [record['avg'][i] for record in logbook]
        min_values = [record['min'][i] for record in logbook]
        
        ax.plot(generations, max_values, 'g-', linewidth=2, label='Max', alpha=0.8)
        ax.plot(generations, avg_values, 'b-', linewidth=2, label='Avg', alpha=0.8)
        ax.plot(generations, min_values, 'r-', linewidth=2, label='Min', alpha=0.8)
        
        ax.set_title('Meta feature: ' + metafeature_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Gen')
        ax.set_ylabel('Fitness_value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotation with final values
        final_max = max_values[-1]
        final_avg = avg_values[-1]
    
    # Hide empty subplots
    for i in range(n_total, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('Convergence plot', fontsize=16, fontweight='bold', y=1.02)
    
    # Save plot
    plt.savefig(file_prefix_name + '_convergence_plot.png', dpi=300, bbox_inches='tight')
    print(f"📊 Convergence plot saved: {file_prefix_name}_convergence_plot.png")
    

def run_split(file, target_column_name, file_prefix_name, meta_features=None, population_size=50, generations=300, seed = None, idx = None):
    """Performs single optimization run and creates convergence plot"""
    if seed is None or idx is None:
        raise ValueError("seed or idx might not be defined")
    
    # Load data
    X = file
    from sklearn.preprocessing import StandardScaler
    cont_cols = []
    for col in X.columns:
        if X[col].nunique() > 10:
            cont_cols.append(col)
    if len(cont_cols) > 0:
        scaler = StandardScaler()
        X[cont_cols] = scaler.fit_transform(X[cont_cols])
    
    # Remove index column (first column) and target
    y = X[target_column_name]
    X = X.drop(columns=[target_column_name])
    
    # Remove first column with indices if it exists
    if X.columns[0] == '0' or X.columns[0] == 'Unnamed: 0':
        X = X.iloc[:, 1:]
    
    # print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    # print("🔄 Starting optimization...")
    
    # Create optimizer
    optimizer = EvolutionarySplitOptimizer(
        X=X, 
        y=y, 
        test_size=0.3,
        population_size=population_size, 
        generations=generations,   
        random_state=seed,    
        meta_features=meta_features,
        file=file,
        idx = idx
    )
    
    # Run optimization
    results = optimizer.optimize(verbose=True)
    
    # Create convergence plot
    # create_convergence_plot(results['logbook'], optimizer.metafeature_names, file_prefix_name)
        
    if results['pareto_front']:
        save_info = optimizer.save_pareto_solutions(
            pareto_front=results['pareto_front'],
            output_dir=f"../data/splitted/{file_prefix_name}/Mfs_based_Split/",
            prefix=f"{file_prefix_name}_solution"
        )
    
    return results


def main(file_name, aim, seeds, keep_size = False):
    # Create directory if not exist
    output_dir = f"../data/splitted/{file_name}/Mfs_based_Split"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    path = f"../data/raw/{file_name}/{file_name}.parquet"

    df = pd.read_parquet(path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

    # Keep 800,000 samples for the sake of speed and efficiency
    if df.shape[0] > 1000000 and not keep_size:
        df = df.sample(n = 800000, random_state = 42) 
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        print("Remove some samples due to extensive size")
        print(f"New Data: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    
    for idx, seed in enumerate(seeds):
        if idx < 6:
            continue
        run_split(
            file=df,
            target_column_name=df.columns[-1],
            file_prefix_name=file_name, #output file name
            meta_features=[aim],
            population_size=10,
            generations=25, 
            seed = seed,
            idx = idx
        )
