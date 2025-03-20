import React, { useContext, useState } from "react";
import { EditedContentContext } from "./AppContext";
import "./LogoSection.css";
import apriori from "../assets/img/apriori.webp";
import kmeans from "../assets/img/kmeans.png";
import decision_tree from "../assets/img/decision_tree.png";
import rough_set from "../assets/img/rough_set.webp";
import naive_bayes from "../assets/img/naive_bayes.png";
import kohonen from "../assets/img/kohonen.png";
import dbscan from "../assets/img/dbscan.png";
import factor_analyzer from "../assets/img/factor_analyzer.png";
import algorithms from "../assets/img/machine_learning.png";
import Apriori from "./Apriori";
import RoughSet from "./RoughSet";
import DecisionTreeGini from "./DecisionTreeGini";
import KMeans from "./KMeans";
import NaiveBayes from "./NaiveBayes";
import Kohonen from "./Kohonen";
import DBSCAN from "./DBSCAN";
import FactorAnalyzer from "./FactorAnalyzer";
const LogoSection = () => {
  const { editedContent } = useContext(EditedContentContext); // Lấy trạng thái toàn cục
  const [selectedAlgorithm, setSelectedAlgorithm] = useState(null); // Trạng thái lưu thuật toán được chọn

  const handleAlgorithmClick = (algorithm) => {
    setSelectedAlgorithm(algorithm); // Cập nhật thuật toán khi người dùng click
  };

  return (
    <section className="logo-section">
  <h4>Hỗ trợ các thuật toán bên dưới</h4>
  <div className="logos">
    <div
      className={`logo-item apriori ${selectedAlgorithm === "apriori" ? "active" : ""}`}
      onClick={() => handleAlgorithmClick("apriori")}
    >
      <img src={apriori} alt="Apriori" />
      <p>Apriori</p>
    </div>
    <div
      className={`logo-item rough_set ${selectedAlgorithm === "rough_set" ? "active" : ""}`}
      onClick={() => handleAlgorithmClick("rough_set")}
    >
      <img src={rough_set} alt="Rough Set" />
      <p>Rough Set</p>
    </div>
    <div
      className={`logo-item decision_tree ${selectedAlgorithm === "decision_tree" ? "active" : ""}`}
      onClick={() => handleAlgorithmClick("decision_tree")}
    >
      <img src={decision_tree} alt="Decision Tree" />
      <p>Decision Tree</p>
    </div>
    <div
      className={`logo-item kmeans ${selectedAlgorithm === "kmeans" ? "active" : ""}`}
      onClick={() => handleAlgorithmClick("kmeans")}
    >
      <img src={kmeans} alt="K-means" />
      <p>K-means</p>
    </div>
    <div
      className={`logo-item naive_bayes ${selectedAlgorithm === "naive_bayes" ? "active" : ""}`}
      onClick={() => handleAlgorithmClick("naive_bayes")}
    >
      <img src={naive_bayes} alt="Naive Bayes" />
      <p>Naive Bayes</p>
    </div>
    <div
      className={`logo-item kohonen ${selectedAlgorithm === "kohonen" ? "active" : ""}`}
      onClick={() => handleAlgorithmClick("kohonen")}
    >
      <img src={kohonen} alt="Kohonen" />
      <p>Kohonen</p>
    </div>
    
    <div
      className={`logo-item dbscan ${selectedAlgorithm === "dbscan" ? "active" : ""}`}
      onClick={() => handleAlgorithmClick("dbscan")}
    >
      <img src={dbscan} alt="DBSCAN" />
      <p>DBSCAN</p>
    </div>
    <div
      className={`logo-item factor_analyzer ${selectedAlgorithm === "factor_analyzer" ? "active" : ""}`}
      onClick={() => handleAlgorithmClick("factor_analyzer")}
    >
      <img src={factor_analyzer} alt="factor_analyzer" />
      <p>Factor Analyzer</p>
    </div>

  </div>

  {/* Hiển thị giao diện Apriori khi chọn */}
  
  {selectedAlgorithm === "apriori" && <Apriori fileContent={editedContent} />}
  {selectedAlgorithm === "rough_set" && <RoughSet fileContent={editedContent} />}
  {selectedAlgorithm === "decision_tree" && <DecisionTreeGini fileContent={editedContent} />}
  {selectedAlgorithm === "kmeans" && <KMeans fileContent={editedContent} />}
  {selectedAlgorithm === "naive_bayes" && <NaiveBayes fileContent={editedContent} />}
  {selectedAlgorithm === "kohonen" && <Kohonen fileContent={editedContent} />}
  {selectedAlgorithm === "dbscan" && <DBSCAN fileContent={editedContent} />}
  {selectedAlgorithm === "factor_analyzer" && <FactorAnalyzer fileContent={editedContent} />}


</section>

  );
};

export default LogoSection;
