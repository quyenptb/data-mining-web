import pandas as pd
import numpy as np
import json
import csv
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import plotly.graph_objects as go
import pydotplus
from io import StringIO
import os
import base64
from collections import defaultdict
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sympy import symbols, Or, And, simplify_logic
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
from sklearn.preprocessing import StandardScaler
from numpy.linalg import LinAlgError




os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"  # Thay đổi đường dẫn phù hợp với hệ thống

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import plotly.graph_objects as go
import io
import base64
import matplotlib.pyplot as plt

def factor_analysis(points, num_factors):
    """
    Thực hiện phân tích nhân tố và trả về loadings, clusters, và giá trị KMO.
    """
    # Kiểm tra phương sai của dữ liệu
    variances = np.var(points, axis=0)
    if any(variances == 0):
        non_zero_variance_indices = np.where(variances > 0)[0]
        points = points[:, non_zero_variance_indices]

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)

    # Tính KMO
    kmo_all, kmo_model = calculate_kmo(pd.DataFrame(points_scaled))

    # Phân tích nhân tố
    fa = FactorAnalyzer(n_factors=num_factors, rotation='varimax')
    fa.fit(pd.DataFrame(points_scaled))

    loadings = fa.loadings_

    # Tìm cụm cho mỗi biến dựa trên tải cao nhất
    loadings_df = pd.DataFrame(
        loadings,
        columns=[f'Factor {i+1}' for i in range(num_factors)],
        index=[f'Variable {i+1}' for i in range(loadings.shape[0])]
    )

    clusters = {
        factor: loadings_df[factor].nlargest(3).index.tolist()
        for factor in loadings_df.columns
    }

    return loadings, clusters, kmo_model

@csrf_exempt
def factor_analysis_view(request):
    if request.method == "POST":
        steps = []  # Danh sách để lưu trữ các bước

        try:
            # Bước 1: Đọc dữ liệu từ request
            steps.append("Đọc dữ liệu từ request.")
            body = json.loads(request.body)
            points_data = body.get("points")  # Dữ liệu điểm
            num_factors = body.get("num_factors")  # Số nhân tố

            # Bước 2: Kiểm tra tính hợp lệ của dữ liệu
            steps.append("Kiểm tra tính hợp lệ của dữ liệu.")
            if not points_data or not num_factors:
                return JsonResponse({"error": "Dữ liệu không hợp lệ."}, status=400)

            # Bước 3: Đọc dữ liệu từ danh sách điểm
            steps.append("Đọc dữ liệu từ danh sách điểm và chuyển đổi thành DataFrame.")
            df = pd.DataFrame(points_data)

            # Bước 4: Lấy dữ liệu cần thiết cho phân tích nhân tố
            steps.append("Lấy dữ liệu cần thiết cho phân tích nhân tố.")
            points = df.values.astype(float)

            # Bước 5: Làm sạch dữ liệu
            steps.append("Làm sạch dữ liệu: thay thế NaN và Inf bằng giá trị trung bình của cột.")
            for i in range(points.shape[1]):
                col = points[:, i]
                col_mean = np.nanmean(col)  # Tính giá trị trung bình, bỏ qua NaN
                col[np.isnan(col)] = col_mean  # Thay thế NaN bằng giá trị trung bình
                col[np.isinf(col)] = col_mean  # Thay thế Inf bằng giá trị trung bình

            # Bước 6: Kiểm tra lại sau khi làm sạch
            steps.append("Kiểm tra lại dữ liệu sau khi làm sạch.")
            if np.isnan(points).any() or np.isinf(points).any():
                return JsonResponse({"error": "Dữ liệu vẫn chứa giá trị không hợp lệ."}, status=400)

            # Bước 7: Thực hiện phân tích nhân tố
            steps.append("Thực hiện phân tích nhân tố.")
            loadings, clusters, kmo_model = factor_analysis(points, num_factors)

            # Bước 8: Tạo đồ họa phân tích nhân tố
            steps.append("Tạo đồ họa phân tích nhân tố.")
            trace_loadings = []
            for i in range(loadings.shape[1]):
                trace_loadings.append(go.Bar(
                    x=df.columns,
                    y=loadings[:, i],
                    name=f'Factor {i+1}'
                ))

            layout = go.Layout(
                title='Factor Analysis Loadings',
                barmode='group',
                xaxis=dict(title='Variables'),
                yaxis=dict(title='Loadings'),
                showlegend=True
            )

            fig = go.Figure(data=trace_loadings, layout=layout)

            # Bước 9: Chuyển đổi đồ họa thành định dạng ảnh base64
            steps.append("Chuyển đổi đồ họa thành định dạng ảnh base64.")
            img_bytes = fig.to_image(format="png")
            img_base64 = base64.b64encode(img_bytes).decode()

            # Bước 10: Định dạng kết quả trả về
            steps.append("Định dạng kết quả trả về.")
            loadings_result = {f'Factor {i+1}': loadings[:, i].tolist() for i in range(loadings.shape[1])}

            return JsonResponse({
                "message": "Phân tích nhân tố đã được thực hiện thành công.",
                "loadings": loadings_result,
                "clusters": clusters,
                "kmo": kmo_model,
                "graph": img_base64,  # Trả về ảnh đồ họa dưới dạng base64
                "steps": steps  # Trả về danh sách các bước đã thực hiện
            }, status=200)

        except Exception as e:
            return JsonResponse({"error": f"Lỗi xử lý: {str(e)}", "steps": steps}, status=500)

def dbscan(points, eps, min_samples):
    steps = []  # Danh sách chứa các bước xử lý
    n = len(points)
    labels = -np.ones(n)  # -1 cho điểm nhiễu
    cluster_id = 0

    # Tính toán khoảng cách giữa các điểm
    def region_query(point_idx):
        distances = np.linalg.norm(points - points[point_idx], axis=1)
        return np.where(distances < eps)[0]

    steps.append("Khởi tạo nhãn cho tất cả các điểm là -1 (điểm nhiễu).")
    steps.append("Bắt đầu phân tích từng điểm trong tập dữ liệu.")

    for point_idx in range(n):
        if labels[point_idx] != -1:  # Nếu đã được gán nhãn
            continue

        neighbors = region_query(point_idx)

        if len(neighbors) < min_samples:  # Điểm nhiễu
            labels[point_idx] = -1
            steps.append(f"Điểm {point_idx} không đủ hàng xóm, gán nhãn là điểm nhiễu.")
        else:
            cluster_id += 1
            labels[point_idx] = cluster_id
            steps.append(f"Bắt đầu một cụm mới với điểm {point_idx}.")

            # Gán nhãn cho các điểm lân cận
            for neighbor_idx in neighbors:
                if labels[neighbor_idx] == -1:  # Nếu là điểm nhiễu
                    labels[neighbor_idx] = cluster_id
                    steps.append(f"Gán nhãn cho điểm nhiễu {neighbor_idx} vào cụm {cluster_id}.")

                if labels[neighbor_idx] == 0:  # Nếu chưa được gán nhãn
                    labels[neighbor_idx] = cluster_id
                    steps.append(f"Gán nhãn cho điểm {neighbor_idx} vào cụm {cluster_id}.")
                    # Tìm thêm các điểm lân cận
                    new_neighbors = region_query(neighbor_idx)
                    if len(new_neighbors) >= min_samples:
                        neighbors = np.append(neighbors, new_neighbors)
                        steps.append(f"Tìm thêm hàng xóm cho điểm {neighbor_idx}.")

    return labels, steps
@csrf_exempt
def dbscan_view(request):
    if request.method == "POST":
        try:
            # Đọc dữ liệu từ request
            body = json.loads(request.body)
            points_data = body.get("points")  # Dữ liệu điểm
            eps = body.get("eps")              # Khoảng cách tối đa
            min_samples = body.get("min_samples")  # Số lượng mẫu tối thiểu

            if not points_data or eps is None or min_samples is None:
                return JsonResponse({"error": "Dữ liệu không hợp lệ."}, status=400)

            # Đọc dữ liệu từ CSV nếu points_data là chuỗi CSV
            df = pd.DataFrame(points_data)

            # Lấy dữ liệu chỉ cần dùng cho thuật toán DBSCAN
            points = df[['gia_tri_1', 'gia_tri_2']].astype(float).values
            names = df['bien'].values.tolist()

            # Thực thi thuật toán DBSCAN
            labels, steps = dbscan(points, eps, min_samples)

            # Tạo đồ họa phân cụm
            trace_points = []
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:
                    color = 'black'  # Màu cho điểm nhiễu
                    label_name = 'Noise'
                else:
                    color = f'hsl({label * 360 / len(unique_labels)}, 100%, 50%)'
                    label_name = f'Cluster {label + 1}'

                cluster_points = points[labels == label]
                trace_points.append(go.Scatter(
                    x=cluster_points[:, 0],
                    y=cluster_points[:, 1],
                    mode='markers',
                    name=label_name,
                    marker=dict(color=color)
                ))

            # Vẽ đồ thị
            layout = go.Layout(
                title='DBSCAN Clustering',
                xaxis=dict(title='x'),
                yaxis=dict(title='y'),
                showlegend=True
            )

            fig = go.Figure(data=trace_points, layout=layout)

            # Chuyển đổi đồ họa thành định dạng ảnh base64
            img_bytes = fig.to_image(format="png")
            img_base64 = base64.b64encode(img_bytes).decode()

            # Định dạng kết quả trả về
            clusters_result = []
            for label in unique_labels:
                cluster_result = {f"Cluster {label + 1}" if label != -1 else "Noise": []}
                for idx in np.where(labels == label)[0]:
                    cluster_result[f"Cluster {label + 1}" if label != -1 else "Noise"].append({
                        "name": names[idx],
                        "coordinates": list(points[idx])
                    })
                clusters_result.append(cluster_result)

            return JsonResponse({
                "message": "DBSCAN clustering đã được thực hiện thành công.",
                "clusters": clusters_result,
                "graph": img_base64,  # Trả về ảnh đồ họa dưới dạng base64
                "steps": steps
            }, status=200)

        except Exception as e:
            return JsonResponse({"error": f"Lỗi xử lý: {str(e)}"}, status=500)

@csrf_exempt
def kohonen_view(request):
    if request.method == "OPTIONS":
        return JsonResponse({"message": "OK"}, status=200)  # Preflight response

    if request.method == "POST":
        try:
            body = json.loads(request.body)
            data = body.get("data")  # Dữ liệu cần huấn luyện
            grid_size = body.get("grid_size", (5, 5))  # Kích thước lưới SOM (mặc định là 5x5)
            learning_rate = body.get("learning_rate", 0.5)  # Tốc độ học
            num_iterations = body.get("num_iterations", 100)  # Số vòng lặp
            radius = max(grid_size) / 2  # Bán kính ban đầu

            steps = []  # Danh sách chứa các bước thực hiện
            
            if not data:
                return JsonResponse({"error": "Dữ liệu không hợp lệ."}, status=400)

            steps.append("Nhận dữ liệu và thiết lập các tham số cho SOM.")

            # Chuyển dữ liệu thành DataFrame
            df = pd.DataFrame(data)

            # Lấy dữ liệu chỉ cần dùng cho thuật toán K-means
            points = df[['gia_tri_1', 'gia_tri_2']].astype(float).values
            names = df['bien'].values.tolist()  # Tên điểm

            # Chuẩn hóa dữ liệu
            scaler = MinMaxScaler()
            points = scaler.fit_transform(points)
            steps.append("Dữ liệu đã được chuẩn hóa.")

            # Khởi tạo trọng số ngẫu nhiên
            weights = np.random.random((grid_size[0], grid_size[1], points.shape[1]))
            steps.append("Trọng số của các nút SOM đã được khởi tạo ngẫu nhiên.")

            # Hàm khoảng cách Euclidean
            def euclidean_distance(a, b):
                return np.linalg.norm(a - b)

            # Hàm Gaussian kernel
            def gaussian_kernel(distance, radius):
                return np.exp(-distance**2 / (2 * (radius**2)))

            steps.append("Bắt đầu quá trình huấn luyện SOM.")

            # Huấn luyện SOM
            for t in range(num_iterations):
                # Giảm bán kính và learning rate
                alpha = learning_rate * (1 - t / num_iterations)
                radius_t = radius * (1 - t / num_iterations)

                for x in points:
                    # Tìm BMU
                    distances = np.linalg.norm(weights - x, axis=2)
                    bmu_idx = np.unravel_index(np.argmin(distances), grid_size)

                    # Cập nhật trọng số BMU và lân cận
                    for i in range(grid_size[0]):
                        for j in range(grid_size[1]):
                            distance_to_bmu = euclidean_distance(np.array([i, j]), bmu_idx)
                            if distance_to_bmu < radius_t:
                                influence = gaussian_kernel(distance_to_bmu, radius_t)
                                weights[i, j] += alpha * influence * (x - weights[i, j])

            steps.append("Quá trình huấn luyện SOM đã hoàn tất.")

            # Gán cụm và lưu BMU cho từng điểm
            clusters = {}
            bmu_indices = []
            
            for idx, x in enumerate(points):
                distances = np.linalg.norm(weights - x, axis=2)
                bmu_idx = np.unravel_index(np.argmin(distances), grid_size)
                bmu_indices.append(bmu_idx)

                # Chuyển đổi bmu_idx sang kiểu tuple của int
                bmu_idx_str = str(bmu_idx)
                
                if bmu_idx_str not in clusters:
                    clusters[bmu_idx_str] = []
                clusters[bmu_idx_str].append({
                    "name": names[idx],  # Lấy tên điểm từ danh sách `names`
                    "coordinates": list(x)  # Toạ độ của điểm
                })

            steps.append("Hoàn tất việc gán cụm cho từng điểm.")

            # Tạo đồ thị tương tác với Plotly
            fig = go.Figure()

            # Vẽ các điểm dữ liệu
            fig.add_trace(go.Scatter(
                x=points[:, 0], y=points[:, 1],
                mode='markers',
                name='Data points',
                marker=dict(color='red', size=8)
            ))

            # Vẽ trọng số các nút trong SOM
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    fig.add_trace(go.Scatter(
                        x=[weights[i, j, 0]], y=[weights[i, j, 1]],
                        mode='markers',
                        name=f"Node ({i}, {j})",
                        marker=dict(color='blue', size=10, symbol='cross')
                    ))

            fig.update_layout(
                title="Self-Organizing Map (SOM)",
                xaxis_title="Feature 1",
                yaxis_title="Feature 2",
                showlegend=True,
                hovermode="closest",
                autosize=True
            )

            # Lưu hình ảnh vào bộ nhớ dưới dạng Base64
            img_stream = io.BytesIO()
            fig.write_image(img_stream, format='png')
            img_stream.seek(0)
            img_base64 = base64.b64encode(img_stream.getvalue()).decode('utf-8')
            
            # Chuyển đổi bmu_indices sang kiểu dữ liệu chuẩn của Python
            bmu_indices = [tuple(map(int, idx)) for idx in bmu_indices]  # Chuyển đổi các chỉ số BMU thành tuple của int

            # Chuyển đổi clusters thành kiểu dữ liệu có thể serialize được
            for bmu_idx_str, points_list in clusters.items():
                for point in points_list:
                    # Chuyển đổi tọa độ và làm tròn đến 3 chữ số thập phân
                    point["coordinates"] = [round(float(coord), 3) for coord in point["coordinates"]]  # Làm tròn tọa độ

            return JsonResponse({
                "message": "SOM đã được huấn luyện thành công.",
                "clusters": clusters,  # Trả về các cụm dữ liệu với tên điểm và tọa độ
                "bmu_indices": bmu_indices,  # Trả về chỉ số BMU cho mỗi điểm dữ liệu
                "weights": weights.tolist(),  # Trả về trọng số của các nút (lưới)
                "grid_size": grid_size,  # Kích thước lưới
                "num_iterations": num_iterations,  # Số vòng lặp
                "learning_rate": learning_rate,  # Tốc độ học
                "radius": radius,  # Bán kính ban đầu
                "som_image": img_base64,  # Trả về hình ảnh Base64 của SOM
                "steps": steps  # Trả về các bước thực hiện
            }, status=200)

        except Exception as e:
            return JsonResponse({"error": f"Lỗi xử lý: {str(e)}"}, status=500)
# Hàm tính khoảng cách Euclid
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

@csrf_exempt
def naive_bayes_view(request):
    if request.method == "OPTIONS":
        return JsonResponse({"message": "OK"}, status=200)  # Preflight response

    if request.method == "POST":
        try:
            body = json.loads(request.body)
            data = body.get("fileContent")  
            features = body.get("features")  # Expected to be a list of column names
            target_column = body.get("target")  # Name of the target column (e.g., 'Play')
            query = body.get("query")

            if not data or not features or not target_column or not query:
                return JsonResponse({"error": "Dữ liệu không hợp lệ."}, status=400)

            # Chuyển đổi fileContent thành DataFrame
            df5 = pd.DataFrame(data)

            # Bước 1: Tính xác suất tiên nghiệm (Prior Probabilities)
            prior_probs = df5[target_column].value_counts(normalize=True)
            steps = {"prior_probabilities": prior_probs.to_dict()}

            conditional_probs = {}
            laplace_smoothing = 1

            # Bước 2: Tính xác suất có điều kiện (Conditional Probabilities)
            for column in features:  # Duyệt qua các tính năng (features)
                conditional_probs[column] = {}
                unique_values = df5[column].unique()  # Lấy các giá trị duy nhất của tính năng
                for feature_value in unique_values:
                    conditional_probs[column][feature_value] = {}
                    for target_value in prior_probs.index:  # Duyệt qua các lớp mục tiêu
                        count = len(df5[(df5[column] == feature_value) & (df5[target_column] == target_value)]) + laplace_smoothing
                        total = len(df5[df5[target_column] == target_value]) + laplace_smoothing * len(unique_values)
                        prob = count / total
                        conditional_probs[column][feature_value][target_value] = prob

            steps["conditional_probabilities"] = conditional_probs

            # Bước 3: Tính xác suất cuối cùng cho mỗi lớp mục tiêu (Posterior Probabilities)
            results = {}
            for target_value in prior_probs.index:
                prob = prior_probs[target_value]
                for feature, value in query.items():
                    if value in conditional_probs[feature]:
                        prob *= conditional_probs[feature][value].get(target_value, laplace_smoothing / (len(df5[df5[target_column] == target_value]) + laplace_smoothing * len(df5[feature].unique())))
                    else:
                        prob *= laplace_smoothing / (len(df5[df5[target_column] == target_value]) + laplace_smoothing * len(df5[feature].unique()))
                results[target_value] = prob

            steps["posterior_probabilities"] = results

            # Bước 4: Dự đoán kết quả (Prediction)
            prediction = max(results, key=results.get)

            return JsonResponse({
                "message": "Naive Bayes đã được thực hiện thành công.",
                "steps": steps,  # Trả về các giá trị trong từng bước
                "prediction": prediction  # Trả về lớp có xác suất cao nhất
            }, status=200)

        except Exception as e:
            return JsonResponse({"error": f"Lỗi xử lý: {str(e)}"}, status=500)

# Hàm K-means
def kmeans(points, centroids, k, max_iter=100):
    for _ in range(max_iter):
        # Gán mỗi điểm vào cụm gần nhất
        clusters = {i: [] for i in range(k)}
        for i, point in enumerate(points):
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(i)

        # Cập nhật các centroid
        new_centroids = np.array([np.mean([points[i] for i in clusters[j]], axis=0) for j in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids, clusters

@csrf_exempt
def kmeans_view(request):
    if request.method == "POST":
        try:
            # Đọc dữ liệu từ request
            body = json.loads(request.body)
            points_data = body.get("points")  # Dữ liệu điểm
            k = body.get("num_clusters")     # Số cụm

            if not points_data or not k:
                return JsonResponse({"error": "Dữ liệu không hợp lệ."}, status=400)

            steps = []  # Khởi tạo danh sách bước thực hiện

            # Đọc dữ liệu từ CSV nếu points_data là chuỗi CSV
            df = pd.DataFrame(points_data)
            steps.append("Đọc dữ liệu từ request và chuyển đổi thành DataFrame.")

            # Lấy dữ liệu chỉ cần dùng cho thuật toán K-means
            points = df[['gia_tri_1', 'gia_tri_2']].astype(float).values
            names = df['bien'].values.tolist()
            steps.append("Lấy các giá trị cần thiết cho K-means: 'gia_tri_1' và 'gia_tri_2'.")

            # Chọn k centroids ban đầu (ngẫu nhiên hoặc sử dụng phương pháp khác)
            centroids = points[:k]
            steps.append("Chọn k centroids ban đầu từ dữ liệu.")

            # Thực thi thuật toán K-means
            final_centroids, final_clusters = kmeans(points, centroids, k)
            steps.append("Thực hiện thuật toán K-means để phân cụm dữ liệu.")

            # Tạo đồ họa phân cụm
            trace_points = []
            for i, cluster in final_clusters.items():
                cluster_points = [points[idx] for idx in cluster]
                trace_points.append(go.Scatter(
                    x=[p[0] for p in cluster_points],
                    y=[p[1] for p in cluster_points],
                    mode='markers',
                    name=f'Cluster {i+1}'
                ))

            # Thêm centroid vào đồ thị
            trace_centroids = go.Scatter(
                x=[centroid[0] for centroid in final_centroids],
                y=[centroid[1] for centroid in final_centroids],
                mode='markers+text',
                marker=dict(color='black', size=10),
                text=[f'Centroid {i+1}' for i in range(k)],
                name='Centroids'
            )

            # Vẽ đồ thị
            layout = go.Layout(
                title='K-means Clustering',
                xaxis=dict(title='x'),
                yaxis=dict(title='y'),
                showlegend=True
            )

            fig = go.Figure(data=trace_points + [trace_centroids], layout=layout)

            # Chuyển đổi đồ họa thành định dạng ảnh base64
            img_bytes = fig.to_image(format="png")
            img_base64 = base64.b64encode(img_bytes).decode()
            steps.append("Chuyển đổi đồ họa phân cụm thành định dạng ảnh base64.")

            # Định dạng kết quả trả về
            clusters_result = []
            for i, cluster in final_clusters.items():
                cluster_result = {f"Cluster {i+1}": []}
                for idx in cluster:
                    cluster_result[f"Cluster {i+1}"].append({
                        "name": names[idx],
                        "coordinates": list(points[idx])
                    })
                clusters_result.append(cluster_result)

            centroids_result = [list(centroid) for centroid in final_centroids]
            steps.append("Tạo kết quả phân cụm và centroids.")

            return JsonResponse({
                "message": "K-means clustering đã được thực hiện thành công.",
                "clusters": clusters_result,
                "centroids": centroids_result,
                "graph": img_base64,  # Trả về ảnh đồ họa dưới dạng base64
                "steps": steps  # Trả về các bước thực hiện
            }, status=200)

        except Exception as e:
            return JsonResponse({"error": f"Lỗi xử lý: {str(e)}"}, status=500)
@csrf_exempt
def decision_tree_view(request):
    if request.method == "POST":
        try:
            body = json.loads(request.body)
            data = body.get("data")
            target_column = body.get("target_column")
            feature_columns = body.get("feature_columns")
            method = body.get("method", "gini") 

            if not data or not target_column or not feature_columns:
                return JsonResponse({"error": "Dữ liệu không hợp lệ."}, status=400)

            steps = []  # Khởi tạo danh sách bước thực hiện

            if method not in ["gini", "entropy"]:
                return JsonResponse({"error": "Phương pháp không hợp lệ. Chỉ chấp nhận 'gini' hoặc 'entropy'."}, status=400)

            # Tạo DataFrame
            df = pd.DataFrame(data)
            steps.append("Tạo DataFrame từ dữ liệu đầu vào.")

            # Kiểm tra cột mục tiêu và đặc trưng tồn tại
            if target_column not in df.columns or any(col not in df.columns for col in feature_columns):
                return JsonResponse({"error": "Cột mục tiêu hoặc đặc trưng không tồn tại."}, status=400)

            steps.append("Kiểm tra sự tồn tại của cột mục tiêu và đặc trưng.")

            # Mã hóa nhãn
            le = LabelEncoder()
            for col in df.columns:
                if df[col].dtype == 'object':  # Chỉ mã hóa các cột kiểu object (chuỗi)
                    df[col] = le.fit_transform(df[col].astype(str))
            steps.append("Mã hóa các nhãn trong DataFrame.")

            # Tách đặc trưng và mục tiêu
            X = df[feature_columns]
            y = df[target_column]
            steps.append("Tách dữ liệu thành đặc trưng (X) và mục tiêu (y).")

            # Huấn luyện mô hình với phương pháp tương ứng (Gini hoặc Entropy)
            clf = DecisionTreeClassifier(criterion=method, max_depth=5, random_state=42)
            clf.fit(X, y)
            steps.append("Huấn luyện mô hình Decision Tree với dữ liệu đã tách.")

            # Xuất cây quyết định thành DOT format
            dot_data = StringIO()
            export_graphviz(
                clf,
                out_file=dot_data,
                feature_names=feature_columns,
                class_names=["No", "Yes"],
                filled=True,
                rounded=True,
                special_characters=True
            )
            steps.append("Xuất cây quyết định sang định dạng DOT.")

            # Chuyển đổi DOT thành biểu đồ bằng pydotplus
            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
            png_data = graph.create_png()
            steps.append("Chuyển đổi dữ liệu DOT thành biểu đồ PNG.")

            # Chuyển đổi ảnh thành chuỗi Base64
            img_base64 = base64.b64encode(png_data).decode('utf-8')
            steps.append("Chuyển đổi ảnh PNG thành chuỗi Base64.")

            # Sử dụng Plotly để hiển thị
            fig = go.Figure()
            fig.add_trace(go.Image(
                source=f"data:image/png;base64,{img_base64}",  # Dùng chuỗi Base64 cho source
                name="Decision Tree",
                hoverinfo="skip",  # Ẩn thông tin hover
                opacity=1.0
            ))
            steps.append("Tạo đối tượng hình ảnh trong Plotly để hiển thị cây quyết định.")

            # Tùy chỉnh layout để cây quyết định đẹp hơn
            fig.update_layout(
                title=f"Cây Quyết Định ({method.capitalize()} Index)",
                title_x=0.5,
                title_font=dict(size=24, color='white', family='Arial'),  # Font chữ hiện đại
                paper_bgcolor='rgb(0, 0, 0)',  # Màu nền tối
                plot_bgcolor='rgba(0,0,0,0)',  # Màu nền trong suốt cho plot
                height=650,
                showlegend=False,
                margin=dict(l=0, r=0, t=50, b=0),
                font=dict(family='Arial', size=14, color='white'),  # Font chữ và màu sắc chữ hiện đại
                colorway=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"],  # Bảng màu tùy chọn
            )
            steps.append("Cập nhật layout của biểu đồ để hiển thị đẹp hơn.")

            # Tạo HTML để hiển thị
            plotly_html = fig.to_html(full_html=False)
            steps.append("Tạo HTML từ đối tượng Plotly để trả về cho client.")

            return JsonResponse({
                "message": "Mô hình Decision Tree đã được huấn luyện thành công.",
                "plotly_html": plotly_html,
                "steps": steps  # Trả về các bước thực hiện
            }, status=200)

        except Exception as e:
            return JsonResponse({"error": f"Lỗi xử lý: {str(e)}"}, status=500)
def convert_itemsets_to_list(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "itemsets" and isinstance(value, (pd.Series, pd.DataFrame)):
                data[key] = value.apply(lambda x: list(x))
            else:
                convert_itemsets_to_list(value)  # Gọi đệ quy cho các giá trị khác
    elif isinstance(data, list):
        for item in data:
            convert_itemsets_to_list(item) 

@csrf_exempt
def apriori_view(request):
    # 1. Xử lý request OPTIONS (preflight)
    if request.method == "OPTIONS":
        return JsonResponse({"message": "OK"}, status=200)

    if request.method == "POST":
        try:
            # 2. Đọc dữ liệu từ body của request
            body = json.loads(request.body)
            data = body.get("data")
            min_support = body.get("min_support")
            min_conf = body.get("min_conf")

            steps = []  # Danh sách lưu các bước xử lý
            
            steps.append("Nhận dữ liệu đầu vào.")
            
            steps.append("Kiểm tra dữ liệu.")

            # 3. Kiểm tra dữ liệu đầu vào
            if not data:
                steps.append("Dữ liệu đầu vào trống hoặc không hợp lệ.")
                return JsonResponse({"error": "Dữ liệu không hợp lệ hoặc trống", "steps": steps}, status=400)
            if min_support is None or not (0 < min_support <= 1):
                steps.append("Giá trị min_support không hợp lệ.")
                return JsonResponse({"error": "min_support phải nằm trong khoảng (0, 1]", "steps": steps}, status=400)
            if min_conf is None or not (0 < min_conf <= 1):
                steps.append("Giá trị min_conf không hợp lệ.")
                return JsonResponse({"error": "min_conf phải nằm trong khoảng (0, 1]", "steps": steps}, status=400)

            

            # 4. Chuyển dữ liệu thành DataFrame
            df = pd.DataFrame(data)
            steps.append("Chuyển đổi dữ liệu thành DataFrame.")
            

            # 5. Kiểm tra sự tồn tại của các cột bắt buộc
            required_columns = {"Mahoadon", "Mahang"}
            if not required_columns.issubset(df.columns):
                missing_columns = required_columns - set(df.columns)
                steps.append("Kiểm tra cột bắt buộc.")
                steps.append("Dữ liệu thiếu các cột bắt buộc: " + ", ".join(missing_columns))
                return JsonResponse({
                    "error": "Dữ liệu thiếu cột bắt buộc",
                    "columns": list(df.columns),
                    "steps": steps
                }, status=400)

            

            # 6. Tạo ma trận nhị phân (Binary Matrix)
            rows = df['Mahoadon'].value_counts().index
            cols = df['Mahang'].value_counts().index
            binary_matrix = pd.DataFrame(0, index=rows, columns=cols)

            for _, row in df.iterrows():
                binary_matrix.loc[row['Mahoadon'], row['Mahang']] = 1

            steps.append("Tạo ma trận nhị phân.")
            

            # 7. Chạy thuật toán Apriori
            frequent_itemsets = apriori(binary_matrix, min_support=min_support, use_colnames=True, verbose=1)
            steps.append("Chạy thuật toán Apriori.")
            

            # 8. Kiểm tra nếu không tìm thấy tập phổ biến
            if frequent_itemsets.empty:
                steps.append("Kiểm tra tập phổ biến.")
                return JsonResponse({"error": "Không tìm thấy tập phổ biến nào với min_support đã chọn", "steps": steps}, status=400)

            steps.append("Xác định được các tập phổ biến.")

            # 9. Tính toán tập phổ biến tối đại
            def find_maximal_itemsets(frequent_itemsets):
                maximal_itemsets = set()
                for itemset in frequent_itemsets:
                    if not any(itemset < other for other in frequent_itemsets):
                        maximal_itemsets.add(itemset)
                return maximal_itemsets

            maximal_itemsets = find_maximal_itemsets(frequent_itemsets['itemsets'])
            steps.append("Tính toán tập phổ biến tối đại.")
            

            # 10. Tính các quy tắc kết hợp
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
            steps.append("Tính toán quy tắc kết hợp.")
            

            # 11. Chuyển đổi các giá trị trong cột 'antecedents' và 'consequents' sang dạng danh sách
            rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x))
            rules["consequents"] = rules["consequents"].apply(lambda x: list(x))

            # 12. Chuyển đổi các giá trị trong cột 'itemsets' của frequent_itemsets thành danh sách
            frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].apply(lambda x: list(x))

            # 13. Chuyển tập phổ biến tối đại từ frozenset sang danh sách
            maximal_itemsets_list = [list(itemset) for itemset in maximal_itemsets]
            
            steps.append("Định dạng kết quả và gửi về.")
            

            # 14. Định dạng kết quả trả về dưới dạng JSON
            rules_list = rules.to_dict(orient="records")
            frequent_itemsets_list = frequent_itemsets.to_dict(orient="records")

            return JsonResponse({
                "rules_list": rules_list,
                "maximal_itemsets": maximal_itemsets_list,
                "frequent_itemsets": frequent_itemsets_list,
                "steps": steps,
            }, safe=False)
        
        except Exception as e:
            # 15. Xử lý ngoại lệ và ghi log lỗi
            return JsonResponse({"error": f"Lỗi xử lý: {str(e)}"}, status=500)

def is_tuple(value):
    return isinstance(value, tuple)
@csrf_exempt
def rough_set_view(request):
    if request.method == "OPTIONS":
        return JsonResponse({"message": "OK"}, status=200)  # Preflight response

    if request.method == "POST":
        try:
            # Phân tích nội dung body của request dưới dạng JSON
            try:
                body = json.loads(request.body)
            except json.JSONDecodeError as e:
                return JsonResponse({"error": f"Lỗi phân tích JSON: {str(e)}"}, status=400)

            steps = []  # Khởi tạo danh sách bước thực hiện

            # Lấy các tham số từ dữ liệu đầu vào
            action = body.get("action")  # Thực hiện hành động nào (approximation, dependency, discernibility)

            steps.append(f"Hành động được yêu cầu: {action}")

            if not action:
                return JsonResponse({"error": "Trường 'action' không được phép trống"}, status=400)

            if action not in ["approximation", "dependency", "discernibility", "reducts"]:
                return JsonResponse({"error": "Hành động không hợp lệ. Các hành động hợp lệ: approximation, dependency, discernibility"}, status=400)

            steps.append("Hành động hợp lệ đã được xác nhận.")

            # Lấy dữ liệu đầu vào (DataFrame)
            df2_data = body.get("df2", [])
            df2 = pd.DataFrame(df2_data)  # Tạo DataFrame từ df2
            steps.append("Đã tạo DataFrame từ dữ liệu đầu vào.")

            if action == "approximation":
                # Lấy tập X và B cho xấp xỉ
                X = set(body.get("X", []))  # Tập X cần tính toán xấp xỉ
                B = body.get("B", [])  # Tập thuộc tính B

                steps.append(f"Tập X được cung cấp: {X}, Tập B được cung cấp: {B}")

                if not X:
                    return JsonResponse({"error": "Tập X không hợp lệ hoặc trống"}, status=400)
                if not B:
                    return JsonResponse({"error": "Tập thuộc tính B không hợp lệ hoặc trống"}, status=400)

                # Tính toán các lớp tương đương
                equivalence_classes = {
                    str(key): list(value) for key, value in df2.groupby(B)["ID"].apply(set).to_dict().items()
                }
                steps.append("Đã tính toán các lớp tương đương dựa trên thuộc tính B.")

                # Tính toán xấp xỉ dưới và xấp xỉ trên
                lower_approximation = set()
                upper_approximation = set()

                for eq_class in equivalence_classes.values():
                    if set(eq_class).issubset(X):
                        lower_approximation.update(eq_class)
                    if set(eq_class) & X:
                        upper_approximation.update(eq_class)

                # Tính toán độ chính xác của xấp xỉ
                accuracy = len(lower_approximation) / len(upper_approximation) if upper_approximation else 0
                steps.append("Đã tính toán xấp xỉ dưới và xấp xỉ trên, cùng với độ chính xác.")

                return JsonResponse({
                    "equivalence_classes": equivalence_classes,
                    "lower_approximation": list(lower_approximation),
                    "upper_approximation": list(upper_approximation),
                    "accuracy": accuracy
                })

            elif action == "dependency":
                # Lấy tập thuộc tính C và B cho khảo sát sự phụ thuộc
                C = body.get("C")  # Tập thuộc tính C (ví dụ: 'Ketqua')
                B = body.get("B")  # Tập thuộc tính B (ví dụ: ['Troi', 'Gio'])

                steps.append(f"Tập C được cung cấp: {C}, Tập B được cung cấp: {B}")

                if not C or not B:
                    return JsonResponse({"error": "Tập thuộc tính C hoặc B không hợp lệ hoặc trống"}, status=400)

                # Khảo sát sự phụ thuộc tính của C vào B
                grouped = df2.groupby(B)
                consistent_count = 0
                total_count = len(df2)

                for _, group in grouped:
                    if len(group[C].unique()) == 1:
                        consistent_count += len(group)

                # Tính toán mức độ phụ thuộc
                gamma_B_C = consistent_count / total_count if total_count else 0
                steps.append("Đã khảo sát sự phụ thuộc và tính toán mức độ phụ thuộc giữa C và B.")

                return JsonResponse({
                    "dependency_degree": gamma_B_C
                })

            elif action == "discernibility":
                attributes = body.get("attributes", [])
                steps.append(f"Các thuộc tính được cung cấp cho tính toán phân biệt: {attributes}")

                if not attributes:
                    return JsonResponse({"error": "Các thuộc tính không hợp lệ hoặc trống"}, status=400)
                if not isinstance(attributes, list):
                    return JsonResponse({"error": "Danh sách attributes phải là một danh sách"}, status=400)
                if not set(attributes).issubset(df2.columns):
                    return JsonResponse({"error": "Một hoặc nhiều thuộc tính trong 'attributes' không tồn tại trong df2"}, status=400)
                if df2.isnull().values.any():
                    return JsonResponse({"error": "Dữ liệu df2 chứa giá trị rỗng (NaN hoặc None), không thể so sánh"}, status=400)

                n = len(df2)
                steps.append("Bắt đầu tính toán ma trận phân biệt cho các thuộc tính được cung cấp.")

                discernibility_matrix = pd.DataFrame(index=df2.index, columns=df2.index, dtype=object)
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            diff_attributes = []
                            for attr in attributes:
                                value_i = df2.iloc[i][attr]
                                value_j = df2.iloc[j][attr]
                                if pd.notna(value_i) and pd.notna(value_j) and value_i != value_j:
                                    diff_attributes.append(attr)
                            discernibility_matrix.iloc[i, j] = diff_attributes if diff_attributes else None

                steps.append("Tính toán ma trận phân biệt đã hoàn tất.")

                discernibility_matrix_json = {
                    i: {j: (v if v is None or isinstance(v, list) else [v])
                        for j, v in row.items()}
                    for i, row in discernibility_matrix.to_dict().items()
                }

                return JsonResponse({
                    "discernibility_matrix": discernibility_matrix_json
                })

            elif action == "reducts":
                attributes = ["Troi", "Gio", "Apsuat"]
                n = len(df2)
                discernibility_matrix = []

                steps.append("Bắt đầu tính toán các thuộc tính phân biệt và xây dựng ma trận phân biệt.")

                for i in range(n):
                    for j in range(i + 1, n):
                        diff_attributes = []
                        for attr in attributes:
                            if df2.loc[i, attr] != df2.loc[j, attr]:
                                diff_attributes.append(attr)
                        if diff_attributes:
                            discernibility_matrix.append(diff_attributes)

                formula = And(*[Or(*[symbols(attr) for attr in attrs]) for attrs in discernibility_matrix])
                steps.append("Đã tạo công thức phân biệt từ ma trận phân biệt.")

                # Rút gọn công thức phân biệt
                simplified_formula = simplify_logic(formula, form='dnf')
                steps.append("Đã rút gọn công thức phân biệt thành dạng đơn giản hơn.")

                return JsonResponse({
                    "simplified_formula": str(simplified_formula)
                })

            else:
                return JsonResponse({"error": "Hành động không hợp lệ"}, status=400)

        except Exception as e:
            print(f"Lỗi: {str(e)}")  # In ra terminal hoặc log file để kiểm tra chi tiết lỗi
            return JsonResponse({"error": f"Lỗi xử lý: {str(e)}"}, status=500)    

