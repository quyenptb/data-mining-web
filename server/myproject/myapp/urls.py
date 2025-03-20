from django.urls import path
from . import views  # Import views từ file views.py

urlpatterns = [
    path('apriori/', views.apriori_view, name='apriori'),  # Đường dẫn tới view apriori_view
    path('rough_set/', views.rough_set_view, name='rough_set'),  # Đường dẫn tới view rough_set_view
    path('decision_tree_view/', views.decision_tree_view, name='decision_tree'),
    path('kmeans/', views.kmeans_view, name='kmeans'), 
    path('naive_bayes/', views.naive_bayes_view, name='naive_bayes'),
    path('kohonen/', views.kohonen_view, name='naive_bayes'),
    path('dbscan/', views.dbscan_view, name='dbscan'),
    path('factor-analyzer/', views.factor_analysis_view, name='factor-analyzer') 
    ]
