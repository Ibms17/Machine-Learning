from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn_lvq import GlvqModel

def home(request):
    return render(request, 'home.html')

def classify(request):
    if request.method == 'POST':
        # Get user inputs
        dataset = request.POST.get('dataset')
        learning_rate = float(request.POST.get('learning_rate'))
        dec_alpha = float(request.POST.get('dec_alpha'))
        min_alpha = float(request.POST.get('min_alpha'))
        max_epoch = int(request.POST.get('max_epoch'))

        # Load dataset
        data = pd.read_csv(dataset)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Feature scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Train LVQ model
        lvq = GlvqModel(prototypes_per_class=1, learning_rate=learning_rate, decay_rate=dec_alpha,
                        random_state=0, batch_size=1, max_iter=max_epoch, learning_decay_rate='invscaling',
                        learning_decay_type='hill')
        lvq.fit(X_train, y_train)

        # Make predictions on test set
        y_pred = lvq.predict(X_test)
        n_zeros = np.sum(y_pred == 0)
        n_ones = np.sum(y_pred == 1)
        acc = accuracy_score(y_test, y_pred)

        # Render results page
        return render(request, 'results.html', {'n_zeros': n_zeros, 'n_ones': n_ones, 'acc': acc})

    else:
        return HttpResponse('Invalid request')