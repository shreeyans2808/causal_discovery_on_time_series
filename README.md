# Time Series Causal Discovery - Comprehensive Evaluation

This project implements and evaluates 17 time series causal discovery algorithms on synthetic and real-world datasets.

## Implemented Algorithms

### Constraint-Based Methods
- **PCMCI**: PC algorithm with momentary conditional independence
- **PCMCI+**: Extension with contemporaneous edges
- **LPCMCI**: Latent PCMCI for handling confounders
- **tsFCI**: Time series Fast Causal Inference
- **CD-NOD**: Causal Discovery from Nonstationary/Heterogeneous Data
- **CDANs**: Causal Discovery from Autocorrelated and Non-stationary data

### FCM-Based Methods
- **VarLiNGAM**: Vector Autoregressive Linear Non-Gaussian Acyclic Model
- **TiMINo**: Time-series Models with Independent Noise

### Gradient-Based Methods
- **DYNOTEARS**: Dynamic NOTEARS for time series
- **NTS-NOTEARS**: Nonparametric Temporal Structure learning

### Granger Causality Methods
- **GVAR**: Generalized Vector AutoRegression
- **NAVAR**: Neural Additive Vector AutoRegression
- **ACD**: Amortized Causal Discovery

### Miscellaneous Methods
- **oCSE**: optimal Causation Entropy
- **TCDF**: Temporal Causal Discovery Framework
- **NBCB**: Noise-based/Constraint-based approach
- **PCTMI**: PC with Temporal Mutual Information

## Installation

```bash
# Create virtual environment (recommended)
python -m venv causal_env
source causal_env/bin/activate  # On Windows: causal_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Additional Setup

Some packages may require additional installation:

```bash
# For lingam (VarLiNGAM)
pip install lingam

# For causal-learn (tsFCI, CD-NOD)
pip install causal-learn

# For tigramite (PCMCI, PCMCI+, LPCMCI)
pip install tigramite
```

## Project Structure

```
.
├── data_generation.py      # Data generation and loading
├── constraint_based.py     # Constraint-based algorithms
├── fcm_based.py           # FCM-based algorithms
├── gradient_based.py      # Gradient-based algorithms
├── granger_misc.py        # Granger & miscellaneous methods
├── evaluation.py          # Evaluation metrics and logging
├── main.py               # Main experiment runner
├── requirements.txt      # Python dependencies
└── fmri_data/           # fMRI simulation data (download separately)
```

## Usage

### Quick Start

Run all experiments:

```bash
python main.py
```

This will:
1. Generate Syn-6 synthetic data
2. Load fMRI data (if available)
3. Run all 17 algorithms on both datasets
4. Evaluate and compare results
5. Save results to log and JSON files

### Individual Algorithm Testing

Test specific algorithm categories:

```bash
# Test constraint-based methods
python constraint_based.py

# Test FCM-based methods
python fcm_based.py

# Test gradient-based methods
python gradient_based.py

# Test Granger & miscellaneous methods
python granger_misc.py
```

### Data Generation Only

```bash
python data_generation.py
```

## Datasets

### Syn-6 Dataset
- **Variables**: 6
- **Lag period**: 2
- **Features**: Autocorrelation, non-stationarity, changing modules
- **Automatically generated**

### fMRI Dataset
- **Source**: BOLD time series simulations
- **Variables**: 50 network nodes
- **Time points**: Variable by simulation
- **Download**: Place .mat files in `./fMRI/` directory

## Output

| Method        | Syn6_SHD | Syn6_TPR | Syn6_FDR | fMRI_SHD | fMRI_TPR | fMRI_FDR | finance_SHD | finance_TPR | finance_FDR |
|---------------|----------|----------|----------|----------|----------|----------|-------------|-------------|-------------|
| ACD           | 51       | 0.429    | 0.94     | 15       | 0.6      | 0.647    | 307         | 0.500       | 0.971       |
| CD-NOD        | 15       | 0.286    | 0.833    | 20       | 0        | 1        | 234         | 0.000       | 1.000       |
| CDANs         | 36       | 1        | 0.837    | 15       | 0.6      | 0.647    | 858         | 1.000       | 0.979       |
| DYNOTEARS     | 7        | 0        | 0        | 10       | 0        | 0        | 18          | 0.000       | 0.000       | 
| GVAR          | 25       | 0        | 1        | 12       | 0.2      | 0.667    | 166         | 0.222       | 0.974       | 
| LPCMCI        | 11       | 1        | 0.611    | 17       | 0.5      | 0.706    |.            |.            |.            |
| NAVAR         | 47       | 0.571    | 0.917    | 12       | 1        | 0.545    | 282         | 0.389       | 0.975       |
| NBCB          | 18       | 1        | 0.72     | 15       | 0.6      | 0.647    | 858         | 1           | 0.979       |
| NTS-NOTEARS   | 12       | 0.286    | 0.778    | 5        | 0.5      | 0        | 18          | 0.000       | 0.000       |
| PCMCI         | 18       | 1        | 0.72     | 15       | 0.6      | 0.647    | 858         | 1.000       | 0.979       |
| PCMCI+        | 8        | 1        | 0.533    | 15       | 0.5      | 0.667    |.            |.            |.            |
| PCTMI         | 31       | 0.857    | 0.833    | 9        | 0.1      | 0        | 582         | 0.000       | 1.000       |
| TCDF          | 25       | 0        | 1        | 11       | 0.2      | 0.6      | 131         | 0.333.      | 0.952       | 
| TiMINo        | 37       | 0.429    | 0.917    | 18       | 0.5      | 0.722    |             |.            |.            |
| VarLiNGAM     | 25       | 0.143    | 0.95     | 18       | 0.2      | 0.833    | 509         | 0.222       | 0.992       | 
| oCSE          | 32       | 0.714    | 0.857    | 7        | 0.3      | 0        | 10          | 0.444       | 0.000       | 

## Dream (Yeast) VarLingam issue as only 21 time steps, same with TCDF (Data generation code says to use only first time series or subject (one subject has only 21 time steps.))

| Method        | Yeast1_SHD | Yeast1__TPR | Yeast1_FDR | Yeast2_SHD | Yeast2_TPR | Yeast2_FDR | Yeast3_SHD | Yeast3_TPR | Yeast3_FDR |
|---------------|------------|-------------|------------|------------|------------|------------|------------|------------|------------|
| ACD           | 4771       | 0.464       | 0.984      | 4880       | 0.450      | 0.964      | 4655       | 0.483      | 0.943      |
| CD-NOD        | 220        | 0.000       | 1.000      | 455        | 0          | 1          | 601        | 0.000      | 1.000      |
| CDANs         | 5578       | 0.072       | 0.998      | 6405       | 0.098      | 0.994      | 6527       | 0.060      | 0.995      |
| DYNOTEARS     | 166        | 0.000       | 0.000      | 389        | 0          | 0          | 551        | 0.000      | 0.000      | 
| GVAR          | 2582       | 0.253       | 0.983      | 2711       | 0.229      | 0.964      | 2767       | 0.258      | 0.943      | 
| LPCMCI        | ..         | .           | .          | ..         | .          | .          |.           |.           |.           |
| NAVAR         | 2832       | 0.337       | 0.980      | 3280       | 0.350      | 0.957      | 3156       | 0.303      | 0.943      |
| NBCB          | 1624       | 0.072       | 0.992      | 1741       | 0.098      | 0.973      | 1811       | 0.060      | 0.975      |
| NTS-NOTEARS   | 166        | 0.000       | 0.000      | 389        | 0          | 0          | 551        | 0.000      | 0.000      |
| PCMCI         | 1624       | 0.072       | 0.992      | 1741       | 0.098      | 0.973      | 1811       | 0.060      | 0.975      |
| PCMCI+        | .          | .           | .....      | ..         | .          | ..         |.           |.           |.           |
| PCTMI         | 6287       | 0.253       | 0.993      | 6574       | 0.404      | 0.976      | 6901       | 0.323      | 0.973      |
| TCDF          | ..         | .           | ..         | ..         | .          | ..         | ..         | .          | ..         | 
| TiMINo        | ..         | .....       | .....      | ..         | .          | ..         |            |.           |.           |
| VarLiNGAM     | ..         | ......      | ....       | ..         | .          | ..         | ..         | .          | ..         | 
| oCSE          | 4230       | 0.398       | 0.984      | 4527       | 0.512      | 0.956      | 4637       | 0.439      | 0.947      | 

## Dream (Ecoli) (same as above for this as well)
| Method        | Ecoli1_SHD | Ecoli1_TPR | Ecoli_FDR | Ecoli2_SHD | Ecoli2_TPR | Ecoli2_FDR |
|---------------|------------|------------|-----------|------------|------------|------------|
| ACD           | 4689       | 0.560      | 0.985     | 4876       | 0.546      | 0.987      |
| CD-NOD        | 175        | 0.000      | 1.000     | 177        | 0          | 1          |
| CDANs         | 6182       | 0.080      | 0.998     | 6255       | 0.050      | 0.999      |
| DYNOTEARS     | 125        | 0          | 0         | 119        | 0          | 0          | 
| GVAR          | 2555       | 0.280      | 0.986     | 2559       | 0.252      | 0.988      | 
| LPCMCI        | ..         | ......     | ....      | ..         | .          | ..         |
| NAVAR         | 3261       | 0.408      | 0.984     | 3004       | 0.462      | 0.982      |
| NBCB          | 1566       | 0.080      | 0.993     | 1349       | 0.050      | 0.995      |
| NTS-NOTEARS   | 125        | 0          | 0         | 119        | 0          | 0          |
| PCMCI         | 1566       | 0.080      | 0.993     | 1349       | 0.050      | 0.995      |
| PCMCI+        | ..         | ......     | ....      | ..         | .          | ..         |
| PCTMI         | 7759       | 0.424      | 0.993     | 6439       | 0.235      | 0.996      |
| TCDF          | ..         | ......     | ....      | ..         | .          | ..         |
| TiMINo        | ..         | ......     | ....      | ..         | .          | ..         |
| VarLiNGAM     | ..         | ......     | ....      | ..         | .          | ..         | 
| oCSE          | 5075       | 0.528      | 0.987     | 4393       | 0.336      | 0.991      | 
### Log Files
- `results_syn6.log`: Detailed results for Syn-6 dataset
- `results_fmri.log`: Detailed results for fMRI dataset
- `results_finance.log`
- `results_dream.log`

### JSON Files
- `results_syn6.json`: Structured results for Syn-6
- `results_fmri.json`: Structured results for fMRI

### Metrics Reported
- **TPR** (True Positive Rate / Recall)
- **FDR** (False Discovery Rate)
- **SHD** (Structural Hamming Distance)
- **Precision**
- **F1 Score**
- **Runtime** (seconds)

## Evaluation Metrics

### True Positive Rate (TPR)
```
TPR = TP / (TP + FN)
```
Higher is better. Measures the proportion of true causal edges correctly identified.

### False Discovery Rate (FDR)
```
FDR = FP / (TP + FP)
```
Lower is better. Measures the proportion of false edges among discovered edges.

### Structural Hamming Distance (SHD)
```
SHD = FP + FN
```
Lower is better. Total number of edge additions and deletions needed to match true graph.

## Customization

### Modify Data Generation
Edit `data_generation.py` to change:
- Number of samples
- Lag period
- Noise distribution
- Structural equations

### Adjust Algorithm Parameters
Each algorithm has configurable parameters:

```python
# Example: Modify PCMCI parameters
method = ConstraintBasedMethods(data, max_lag=3)
result = method.run_pcmci(pc_alpha=0.01, cond_ind_test='gpdc')
```

### Add Custom Algorithms
Create a new method class and add to `main.py`:

```python
class CustomMethod:
    def __init__(self, data, max_lag=2):
        self.data = data
        self.max_lag = max_lag
    
    def run(self):
        # Your implementation
        graph = ...
        return {'graph': graph, 'algorithm': 'Custom'}
```

## Troubleshooting

### Missing Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### GPU Support (Optional)
For faster gradient-based methods:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
Reduce data size or run algorithms individually:
```python
# In main.py, comment out algorithms
algorithms = [
    ('PCMCI', self._run_pcmci),
    # ('PCMCI+', self._run_pcmci_plus),  # Commented out
    # ...
]
```

## References

See the survey papers for detailed algorithm descriptions:
- Hasan et al. (2023): "A Survey on Causal Discovery Methods for I.I.D. and Time Series Data"
- Ferdous et al. (2023): "CDANs: Temporal Causal Discovery from Autocorrelated and Non-Stationary Time Series Data"

## Citation

If you use this code, please cite the original papers for each algorithm you use.

## License

This project is for research and educational purposes.
