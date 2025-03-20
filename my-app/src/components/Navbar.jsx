import React, { useState } from "react";
import "./Navbar.css";
import logo from "../assets/img/logo.png";
import Introduction from "./Introduction";

const Navbar = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const openModal = () => setIsModalOpen(true);
  const closeModal = () => setIsModalOpen(false);

  const handleOverlayClick = (e) => {
    // Đóng modal nếu nhấn vào overlay bên ngoài modal-content
    if (e.target.classList.contains("modal-overlay")) {
      closeModal();
    }
  };

  return (
    <header>
      <nav className="navbar">
        <div className="center-content">
          <img src={logo} alt="Logo" className="logo" />
          <span className="title">Data Mining</span>
        </div>
        <button className="btn-intro" onClick={openModal}>
          Giới thiệu
        </button>
      </nav>

      {isModalOpen && (
        <div className="modal-overlay" onClick={handleOverlayClick}>
          <div className="modal-content">
            <button className="btn-close" onClick={closeModal}>
              ×
            </button>
            <Introduction />
          </div>
        </div>
      )}
    </header>
  );
};

export default Navbar;
