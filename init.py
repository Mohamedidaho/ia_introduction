import numpy as np
import pandas as pd

import os

# ACQUISITION DES DONNÉES
for dirname, _,filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        filename = os.path.join(dirname, filename)
        print(filename)

from sklearn import svm # séparation
from sklearn.model_selection import train_test_split # apprentissage 
from sklearn.preprocessing import StandardScaler # for feature scaling
from sklearn.model_selection import GridSearchCV # for fine-tuning
from sklearn.metrics import make_scorer, balanced_accuracy_score # for evaluation
from sklearn.pipeline import make_pipeline # for prediction

from scipy import stats
from fitter import Fitter
import copy

import matplotlib.pyplot as plt
import seaborn as sns 

plt.style.use('dark_background')

from pylab import rcParams

rcParams['figure.figsize'] = 10, 5

# Nettoyage des données
df = pd.read_csv("kaggle/input/nba_games/games.csv")
df.head()

# Trie des dataframe par date
df = df.sort_values(by='GAME_DATE_EST').reset_index(drop=True)

#supprimer les entrées vides, les données avant 2004 contiennent NaN
df = df.loc[df['GAME_DATE_EST']>= "2004-01-01"].reset_index(drop=True)

# on verifie qu'on a pas des entrées vides
df.isnull().values.any()

# remplace les ID par les noms d'équipes
df_names = pd.read_csv('kaggle/input/nba_games/teams.csv')
df_names = df_names.head()

# on remplace les colonnes 'HOME_TEAM_ID' et 'VISITOR_TEAM_ID'
df_names = df_names[['TEAM_ID', 'NICKNAME']]

# on remplace 'HOME_TEAM_ID' par 'TEAM_ID' et 'NICKNAME'
home_names = df_names.copy()

#on change le nom de la colonne
home_names.columns = ['HOME_TEAM_ID', 'NICKNAME']

#fusionner les noms en fonction de l'ID
result = pd.merge(df['HOME_TEAM_ID'], home_names, how='left', on='HOME_TEAM_ID')
df['HOME_TEAM_ID'] = result['NICKNAME']

#on remplace 'VISITOR_TEAM_ID' par df_names
visitor_names = df_names.copy()
visitor_names.columns = (['VISITOR_TEAM_ID','NICKNAME'])

#change le nom de la colonne avant fusion
#fusion avec df en fonction de l'ID
result_ = pd.merge(df['VISITOR_TEAM_ID'], visitor_names, how="left", on="VISITOR_TEAM_ID")
df['VISITOR_TEAM_ID'] =  result_['NICKNAME']

#Dataframe final
df.head()


#on veut prédire les resultats des playoff a partir de 2020-08
df = df.loc[df['GAME_DATE_EST']<'2020-08-01'].reset_index(drop=True)

feature_list = list(df.columns)

#On sélectionne les champs de comparaison pour la simulation
selected_features = [
    'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home',
    'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away',
    ]

X = df[selected_features]
X.head()

#on verifie nos cibles
y= df['HOME_TEAM_WINS']
y.head()

#on convertie nos données en tableau de valeur
X = X.to_numpy()
y = y.to_numpy()

#Mis en place du SVM

#training
#Divise les matrices en sous-ensembles de train 
# et de test aléatoires
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.3, random_state=42
)
print("X shape: ",X_train.shape, "y shape: ",y_train.shape)

#Training SVM
clf = svm.SVC(kernel="linear")
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print('score précision: ',balanced_accuracy_score(y_test,y_pred))

#Affinement des paramètre
time
#Stratégie pour évaluer les performances du modèle à validation croisée sur l'ensemble de test
#Crée un marqueur à partir d'une métrique de performance ou d'une fonction de perte
scoring = make_scorer(balanced_accuracy_score) #

# Dictionnaire avec des clés 
# et des listes de réglages de paramètres à essayer comme valeurs, 
# ou une liste de ces dictionnaires
#Cela permet de faire une recherche sur n'importe quelle séquence de réglages de paramètres
param_grid = { 'C':[0.1,1,10],
                'gamma' : [1,0.1,0.01]
}

#Recherche exhaustive sur des valeurs de paramètre spécifiées pour l'estimateur
grid = GridSearchCV(svm.SVC(kernel='linear'),param_grid,scoring=scoring,refit=True,verbose=2) # refit -> Réajuster un estimateur en utilisant les meilleurs paramètres trouvés sur l'ensemble de données 
grid.fit(X_train,y_train)

#On affiche les meilleurs hyperparamètres pour le modele
Dis = grid.best_estimator_
print(Dis)

#mis en place du generateur
#on filtre nos données pour retenir les plus récentes
df_ = df.loc[df['GAME_DATE_EST']>'2019-10-01'].reset_index(drop=True)
df_.head()

#on defini la liste des distribution 
#pour le montage
selected_distributions = [
    'norm','t', 'f', 'chi', 'cosine', 'alpha', 
    'beta', 'gamma', 'dgamma', 'dweibull',
    'maxwell', 'pareto', 'fisk'
]

#extraction des équipes uniques
unique_teams = df['HOME_TEAM_ID'].unique()


#on conbine les données pour tous les matchs
all_team_sim_data = {}

#on cheche les matchs disputés par l'equipe
for team_name in unique_teams:
    df_team = df_.loc[(df_['HOME_TEAM_ID'] == team_name) | (df_['VISITOR_TEAM_ID'] == team_name)]
    
    #si l'ekip accueille on selectionne les 5 premiers caractéristiques
    df_1 = df_team.loc[df_team['HOME_TEAM_ID'] == team_name][selected_features[:5]]

    #si l'ekip est visiteuse, on selectionne les 5 derniers
    df_0 = df_team.loc[df_team['VISITOR_TEAM_ID'] == team_name][selected_features[5:]]

    #on combine le tout
    df_0.columns = df_1.columns #on fait matcher les colonnes semblables
    df_s = pd.concat([df_1,df_0], axis=0)

    #conversion du DF en numpy array
    all_team_sim_data[team_name] = df_s.to_numpy()


megadata = {}
for team_name in unique_teams:
    feauture_dis_paras = []
    data = all_team_sim_data[team_name]


    for i in range(5):
        f = Fitter(data)
        f.distributions = selected_distributions

        f.fit()
        best_paras = f.get_best(method='sumsquare_error')

        feauture_dis_paras.append(best_paras)

    megadata[team_name] = feauture_dis_paras
print('Les caractéristiques de toutes les équipes ont été chargés')
print(megadata)

# SIMULATION
DATA = megadata.copy()

GEN = {
    'alpha': stats.alpha.rvs,
    'beta': stats.beta.rvs,
    'chi': stats.chi.rvs,
    'cosine': stats.cosine.rvs,
    'dgamma': stats.dgamma.rvs,
    'dweibull':stats.dweibull.rvs,
    'f':stats.f.rvs,
    'fisk':stats.fisk.rvs,
    'gamma': stats.gamma.rvs,
    'maxwell':stats.maxwell.rvs,
    'norm':stats.norm.rvs,
    'pareto':stats.pareto.rvs,
    't':stats.t.rvs,
}

DIS = make_pipeline(scaler,Dis)

class Game:
    def __init__(self, random_state = None):
        self.random_state = random_state

    def predict(self,team1,team2, num_games = 1):
        assert num_games >= 1,"au moins un match doit être jouer"
        team_1_feature_data = DATA[team1]
        team_2_feature_data = DATA[team2]
        features = []
        
        for feature_paras_1 in team_1_feature_data:
            sample_1 = self.sampling(feature_paras_1, size = num_games)
            features.append(sample_1)

        for feature_paras_2 in team_2_feature_data:
            sample_2 = self.sampling(feature_paras_2)
            features.append(sample_2)

        features = np.array(features).T
        win_loss = DIS.predict(features)

        return list(win_loss)

    def sampling(self,dic,size:1,random_state=None):
        dis_name = list(dic.keys()[0])
        paras = list(dic.values()[0])
        sample = GEN[dis_name](*paras,size=size,random_state=random_state)

        return sample

class FinalTournament(Game):
        def __init__(self,n_games_per_group=7,winning_threshold=4,random_state=None):
            self.n_games_per_group = n_games_per_group
            self.winning_threshold =winning_threshold
            self.team_list = {}

            super().__init__(random_state)

        def simulate(self,group_list,n_simulation = 1,probs=T):
            self.rounds = {}
            self.team_list = [i[0]for i in group_list] + [i[1] for i in group_list]

            for i in range(n_simulation):
                cham = self.one_time_simu(group_list)

            if probs:
                self.rounds_probs = self._compute_probs()

        def one_time_simu(self,group_list,verbose=False,probs=False):
            if self.team_list == None:
                self.team_List = [i[0] for i in group_list] + [i[1] for i in group_list]
                round_number, done = 0, 0
                while not done:
                    all_group_winners, group_list = self.play_round(group_list)
                    try:
                        updated_round_stats = self.rounds[round_number]
                    except KeyError:
                        updated_round_stats = {}
                        for team in self.team_list:
                            updated_round_stats[team] = 0

                    for winner in all_group_winners:
                        try:
                            updated_round_stats[team] =+1
                        except KeyError:
                            pass
                    self.rounds[round_number] = updated_round_stats
                    if verbose:
                        print('{} tour joué'.formt(round_number))
                    if probs:
                        self.rounds_probs = self._compute_probs()
                    if type(group_list) != list:
                        done = 1
                        round_number=+1
            
            return group_list
        
        def play_round(self, group_list):
            all_group_winners = []
            for group in group_list:
                winner = self.play_n_games(group[0],group[1])
                all_group_winners.append(winner)

            if len(all_group_winners) > 1:
                new_group_list = []
                for index in range(0,len(all_group_winners),2):
                    new_group = [all_group_winners[index]],all_group_winners[index+1]]
                    new_group_list.append(new_group)
                return all_group_winners, new_group_list
            else:
                return all_group_winners,winner

        def play_n_games(self,team1,team2):
            result = Game().predict(team1,team2,self.n_games_per_group)
            if sum(result[:4]) == self.winning_threshold or sum(result)>=self.winning_threshold:
                winner = team1
            else:
                winner = team2

            return winner

        
        def _compute_probs(self):
            round_probs = copy.deepcopy(self.rounds)
            for round_number, round_stats in round_probs.items():
                m = np.sum(list(round_stats.values()))
                for k,v in round_probs[round_number].items():
                    round_probs[round_number][k] = v/m
            return round_probs

group_list = [
     # Eastern Conference
     ('Bucks', 'Magic'),  # group A 
     ('Pacers', 'Heat'), # group B
    
     ('Celtics', '76ers'), # group C
     ('Raptors', 'Nets'), # group D
    
     # Western Conference
     ('Lakers','Trail Blazers'),  # group E
     ('Rockets','Thunder'), # group F
    
     ('Nuggets', 'Jazz'), # group G
     ('Clippers', 'Mavericks')] # group H      
    

# initiate a playoff
playoff = FinalTournament()
# simulate the playoff 5,000 times
playoff.simulate(group_list, n_simulation = 5000)

playoff.rounds_probs