import React from "react";
import "./Introduction.css";
import book from "../assets/img/book.png";

const Introduction = () => {
  return (
    <div>

      <div id="introduction" className="introduction">
        <div className="intro-container">
          <h1 className="intro-title">
            Đồ án môn học <br></br>
            <span>Khai thác dữ liệu</span>
          </h1>
          <p className="intro-subtitle">
            Lớp <span>IS252.P11</span>, Thầy <span>Mai Xuân Hùng</span>
          </p>

          <div className="intro-team-section">
            <h2>Thành viên nhóm 5</h2>
            <ul>
              <li href="#">
                <a href="https://example.com" > 
                Nguyễn Lê Vy - 21522811 </a></li>
              <li href="#">
                <a href="https://example.com"> 
                Phan Thị Bích Quyên - 22521224
                </a>
                </li>
            </ul>
          </div>

          <div className="intro-resources">
            <h2>Tài liệu tham khảo</h2>
            <ul>
              <li>
                <a href="https://example.com" target="_blank">
                  Tài liệu đồ án
                </a>
              </li>
              <li>
                <a href="https://example.com" target="_blank">
                  Hướng dẫn sử dụng
                </a>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Introduction;
