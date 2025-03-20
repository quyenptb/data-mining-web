import React, { useState } from "react";
import Step from "./Step";
import FileContent from "./FileContent";
import Papa from "papaparse";
import * as XLSX from "xlsx";
import "./GuideSection.css";
import uploadFile from "../assets/img/uploadFile.png";
import chooseAlgo from "../assets/img/chooseAlgo.png";
import watchResult from "../assets/img/watchResult.png";

const GuideSection = () => {
  const [fileContent, setFileContent] = useState(null);
  const [fileError, setFileError] = useState(null);
  const [fileName, setFileName] = useState(""); // Lưu tên file

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setFileError(null); // Reset lỗi trước đó

    if (file) {
      setFileName(file.name); // Lưu tên file
      const ext = file.name.split(".").pop().toLowerCase();

      if (file.size > 5 * 1024 * 1024) {
        setFileError("File quá lớn. Vui lòng tải lên file có kích thước nhỏ hơn 5MB.");
        return;
      }

      const reader = new FileReader();

      if (ext === "csv") {
        reader.onload = () => {
          Papa.parse(reader.result, {
            complete: (result) => {
              setFileContent(result.data);
            },
            header: true,
          });
        };
        reader.readAsText(file);
      } else if (ext === "xlsx") {
        reader.onload = () => {
          const wb = XLSX.read(reader.result, { type: "binary" });
          const ws = wb.Sheets[wb.SheetNames[0]];
          const json = XLSX.utils.sheet_to_json(ws);
          setFileContent(json);
        };
        reader.readAsBinaryString(file);
      } else {
        setFileError("Định dạng file không được hỗ trợ. Vui lòng tải lên file .csv hoặc .xlsx.");
      }
    }
  };

  const handleIconClick = () => {
    document.getElementById("fileInput").click(); // Click vào input ẩn
  };

  return (
    <section className="guide-section">
      <h1>Hướng dẫn sử dụng</h1>
      <p className="subtitle">3 bước</p>
      <div className="steps">
        {/* Bước 1: Tải lên file */}
        <div className="step-container">
          <div onClick={handleIconClick} className="upload-icon-container">
            <Step
              icon={uploadFile}
              title="Tải lên file"
              description="Hỗ trợ hai loại file là .csv và file excel. Kích thước không quá 5MB."
            />
          </div>
          <input
            id="fileInput"
            type="file"
            accept=".csv, .xlsx"
            onChange={handleFileChange}
            style={{ display: "none" }} // Ẩn input
          />
          {fileError && <p className="error-message">{fileError}</p>}
        </div>

        {/* Bước 2: Chọn thuật toán */}
        <Step
          icon={chooseAlgo}
          title="Chọn thuật toán"
          description="Trang web hỗ trợ nhiều loại thuật toán trong lĩnh vực Data mining."
        />

        {/* Bước 3: Xem kết quả */}
        <Step
          icon={watchResult}
          title="Xem kết quả"
          description="Kết quả được trả về nhanh chóng, chính xác."
        />
      </div>

      {/* Hiển thị nội dung file nếu có */}
      {fileContent && <FileContent fileContent={fileContent} fileType={"csv"} />}
    </section>
  );
};

export default GuideSection;
