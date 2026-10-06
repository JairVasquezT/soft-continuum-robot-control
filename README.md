# soft-continuum-robot-control

[English](#english) · [Français](#français)

---

## English

Learning-based predictive control and data pipeline for a **cable-driven soft continuum manipulator** (4 motors, 3-DoF end-effector position). The controller combines an LSTM predictor with a Cross-Entropy Method (CEM) optimizer and a differentiable Jacobian anchor, and runs in closed loop on the real robot.

Developed during a research internship at ISIR (Sorbonne Université). The associated manuscript is **under review**.

### Demo video

<!-- Add the demo video link here, e.g. [Watch the video](https://...) -->

### What the code does

1. **Data collection** (`main.py`, `hardware/`, `utils/`): drives the Dynamixel motors through generated command combinations while recording motor positions, OptiTrack poses and load-cell forces to CSV.
2. **Dataset building and training** (`ml/`): turns the logs into filtered, normalized datasets (`dataset_*.py`), and trains the models (`train_*.py`). Two networks are used:
   - an **observer** LSTM (`SoftRobotLSTM`, in `control/MPC.py`) that estimates the state from recent history;
   - a **predictor** (`MPCDirectPredictor` in `control/MPC.py`; encoder–decoder variant in `ml/predicteur.py`) that predicts the end-effector trajectory for candidate motor commands.
3. **Control** (`control/MPC.py`, `control/candidate.py`): at each step the controller samples candidate commands, scores them with the batched predictor, and keeps the best. Three optimization modes are available through `--modo-control`:
   - `cem`: Cross-Entropy Method with multi-start only;
   - `jacobiano`: local inverse kinematics using the Jacobian of the predictor (autograd), re-linearized `--k-iters-jacobiano` times;
   - `hibrido` (default): the Jacobian solution is added as one more anchor of the CEM multi-start.

### Repository layout

```
.
├── main.py                 # Data collection with Dynamixel + OptiTrack (+ load cells)
├── go_home.py              # Sends all motors to the calibrated home position
├── config/robot_config.py  # Motor IDs, home positions, limits, sampling grid
├── control/                # Closed-loop MPC and experiment analysis
│   ├── MPC.py              #   Main controller / experiment runner (entry point)
│   ├── candidate.py        #   Candidate generator, cost function, CEM controller
│   ├── trajectories.py     #   Command combinations and waypoint generation
│   ├── analizar_experimento.py, plot_resultados.py, calc_hz.py, result_3d.py
│   │                       #   Analysis and plotting of experiment CSVs
│   ├── diagnostico_*.py    #   Calibration / observer diagnostics
│   └── dataset_*_params.json   # Normalization metadata read by MPC.py
├── ml/                     # Dataset generation, training and validation
│   ├── dataset_*.py        #   Dataset builders (versions v01–v16, see below)
│   ├── train_*.py          #   Training scripts
│   ├── run_experimentos_*.py   # Sweeps (window size, loss weights)
│   ├── logs_experimentos/  #   Text logs of training sweeps
│   ├── dataset_*_params.json   # Normalization metadata of each dataset version
│   └── prueba/             #   Older / scratch copies of dataset metadata
├── hardware/               # Dynamixel, Phidget load cells, OptiTrack (NatNet) clients
├── utils/                  # Motor calibration routines
├── examples/               # Small OptiTrack + Dynamixel sync example
└── docs/
    ├── figures/            #   Result figures (3D result, trajectory/error plots)
    └── results/            #   Saved text outputs of evaluations
```

**Dataset versions.** The name encodes the inputs used: `real` / `completo` (data source), `meta` / `torque` / `tension` (extra input signals), and `sinRot` / `conRot` (without / with end-effector orientation). The `_params.json` file stores the normalization parameters and must match the model weights.

### Hardware

- Dynamixel EX-106+ motors (4), IDs and home positions in `config/robot_config.py`
- OptiTrack motion capture (Motive, NatNet) as the position source
- Phidget load cells (4 channels) for cable tension

### Setup

The code imports itself as the package `continuum_robot`, so **clone it into a folder with that name** and run scripts from its parent folder:

```bash
git clone https://github.com/JairVasquezT/soft-continuum-robot-control.git continuum_robot
cd continuum_robot
pip install -r requirements.txt
```

Model weights (`*.pth`), `.npy` datasets and `.csv` logs are **not** stored in the repository (see `.gitignore`).

### Running

```bash
# From the folder that contains continuum_robot/

# Simulated hardware (no robot needed)
python continuum_robot/control/MPC.py

# Real robot with OptiTrack, hybrid CEM + Jacobian controller
python continuum_robot/control/MPC.py --no-simulate --use-optitrack --modo-control hibrido --k-iters-jacobiano 3

# Repeated-trial experiment over the waypoints defined in MPC.py
python continuum_robot/control/MPC.py --no-simulate --use-optitrack --experimento-repeticiones --n-repeticiones 5

# Send motors to home
python continuum_robot/go_home.py
```

`control/MPC.py` expects the following files in `control/`: the predictor and observer weights (`*.pth`) and the two `dataset_*_params.json` files whose names are set at the end of the script.

Run `python continuum_robot/control/MPC.py --help` for all options.

### Status and license

Research code, provided as is. The manuscript is under review. See `LICENSE`.

---

## Français

Pipeline de données et contrôle prédictif par apprentissage pour un **manipulateur continu souple actionné par câbles** (4 moteurs, position de l'effecteur en 3D). Le contrôleur combine un prédicteur LSTM, un optimiseur par méthode d'entropie croisée (CEM) et une ancre Jacobienne différentiable, et fonctionne en boucle fermée sur le robot réel.

Développé lors d'un stage de recherche à l'ISIR (Sorbonne Université). L'article associé est **en cours d'évaluation**.

### Vidéo de démonstration

<!-- Ajouter ici le lien de la vidéo, par ex. [Voir la vidéo](https://...) -->

### Ce que fait le code

1. **Collecte de données** (`main.py`, `hardware/`, `utils/`) : pilote les moteurs Dynamixel à travers des combinaisons de commandes générées, en enregistrant positions des moteurs, poses OptiTrack et forces des cellules de charge dans des CSV.
2. **Construction des jeux de données et entraînement** (`ml/`) : transforme les logs en jeux de données filtrés et normalisés (`dataset_*.py`) et entraîne les modèles (`train_*.py`). Deux réseaux sont utilisés :
   - un **observateur** LSTM (`SoftRobotLSTM`, dans `control/MPC.py`) qui estime l'état à partir de l'historique récent ;
   - un **prédicteur** (`MPCDirectPredictor` dans `control/MPC.py` ; variante encodeur–décodeur dans `ml/predicteur.py`) qui prédit la trajectoire de l'effecteur pour des commandes candidates.
3. **Contrôle** (`control/MPC.py`, `control/candidate.py`) : à chaque pas, le contrôleur échantillonne des commandes candidates, les évalue avec le prédicteur par lot et garde la meilleure. Trois modes via `--modo-control` :
   - `cem` : méthode d'entropie croisée avec multi-démarrage uniquement ;
   - `jacobiano` : cinématique inverse locale avec le Jacobien du prédicteur (autograd), re-linéarisée `--k-iters-jacobiano` fois ;
   - `hibrido` (par défaut) : la solution Jacobienne est ajoutée comme une ancre de plus au multi-démarrage du CEM.

### Organisation du dépôt

```
.
├── main.py                 # Collecte de données Dynamixel + OptiTrack (+ cellules de charge)
├── go_home.py              # Envoie tous les moteurs à la position home calibrée
├── config/robot_config.py  # IDs des moteurs, positions home, limites, grille d'échantillonnage
├── control/                # MPC en boucle fermée et analyse des expériences
│   ├── MPC.py              #   Contrôleur principal / expériences (point d'entrée)
│   ├── candidate.py        #   Générateur de candidats, fonction de coût, contrôleur CEM
│   ├── trajectories.py     #   Combinaisons de commandes et génération de waypoints
│   ├── analizar_experimento.py, plot_resultados.py, calc_hz.py, result_3d.py
│   │                       #   Analyse et tracé des CSV d'expériences
│   ├── diagnostico_*.py    #   Diagnostics de calibration / observateur
│   └── dataset_*_params.json   # Métadonnées de normalisation lues par MPC.py
├── ml/                     # Génération de jeux de données, entraînement et validation
│   ├── dataset_*.py        #   Constructeurs de jeux de données (versions v01–v16)
│   ├── train_*.py          #   Scripts d'entraînement
│   ├── run_experimentos_*.py   # Balayages (taille de fenêtre, poids de la loss)
│   ├── logs_experimentos/  #   Logs texte des balayages
│   ├── dataset_*_params.json   # Métadonnées de normalisation de chaque version
│   └── prueba/             #   Anciennes copies / brouillons de métadonnées
├── hardware/               # Clients Dynamixel, cellules Phidget, OptiTrack (NatNet)
├── utils/                  # Routines de calibration des moteurs
├── examples/               # Petit exemple de synchronisation OptiTrack + Dynamixel
└── docs/
    ├── figures/            #   Figures de résultats (résultat 3D, trajectoires/erreurs)
    └── results/            #   Sorties texte d'évaluations
```

**Versions des jeux de données.** Le nom indique les entrées utilisées : `real` / `completo` (source des données), `meta` / `torque` / `tension` (signaux d'entrée supplémentaires) et `sinRot` / `conRot` (sans / avec orientation de l'effecteur). Le fichier `_params.json` contient les paramètres de normalisation et doit correspondre aux poids du modèle.

### Matériel

- 4 moteurs Dynamixel EX-106+, IDs et positions home dans `config/robot_config.py`
- Capture de mouvement OptiTrack (Motive, NatNet) comme source de position
- Cellules de charge Phidget (4 canaux) pour la tension des câbles

### Installation

Le code s'importe lui-même comme le paquet `continuum_robot` : **clonez-le dans un dossier de ce nom** et lancez les scripts depuis le dossier parent :

```bash
git clone https://github.com/JairVasquezT/soft-continuum-robot-control.git continuum_robot
cd continuum_robot
pip install -r requirements.txt
```

Les poids des modèles (`*.pth`), les jeux de données `.npy` et les logs `.csv` **ne sont pas** dans le dépôt (voir `.gitignore`).

### Exécution

```bash
# Depuis le dossier qui contient continuum_robot/

# Matériel simulé (sans robot)
python continuum_robot/control/MPC.py

# Robot réel avec OptiTrack, contrôleur hybride CEM + Jacobien
python continuum_robot/control/MPC.py --no-simulate --use-optitrack --modo-control hibrido --k-iters-jacobiano 3

# Expérience à essais répétés sur les waypoints définis dans MPC.py
python continuum_robot/control/MPC.py --no-simulate --use-optitrack --experimento-repeticiones --n-repeticiones 5

# Envoyer les moteurs à la position home
python continuum_robot/go_home.py
```

`control/MPC.py` attend dans `control/` : les poids du prédicteur et de l'observateur (`*.pth`) et les deux fichiers `dataset_*_params.json` dont les noms sont définis à la fin du script.

Lancez `python continuum_robot/control/MPC.py --help` pour toutes les options.

### Statut et licence

Code de recherche, fourni tel quel. L'article est en cours d'évaluation. Voir `LICENSE`.
