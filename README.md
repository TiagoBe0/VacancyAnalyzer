# VacancyAnalyzer

Vacancy Analizer is a Python-based tool designed for the analysis and prediction of vacancies (defects) in nanomaterials. The program leverages simulation data processed through OVITO to separate clusters, compute key features (such as surface area, particle count, and spatial metrics like maximum and minimum distances to the center of mass), and apply machine learning techniques—specifically, Random Forest regression—to predict the number of vacancies in each cluster.

Key features include:

    Cluster Analysis: Segregates simulation data into clusters and computes geometric properties.
    Feature Engineering: Extracts relevant characteristics (e.g., surface area, neighbor count, max/min distances) to feed into the predictive model.
    Vacancy Prediction: Utilizes Random Forest regression to estimate vacancy counts based on computed features.
    Single Vacancy Filtering: Automatically detects and filters clusters that match predefined criteria for single vacancies, ensuring accurate overall vacancy counting.

This tool is especially useful for researchers in materials science and nanotechnology looking to analyze defect structures in simulated or experimental datasets.
