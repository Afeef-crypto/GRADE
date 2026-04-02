import { NavLink, Route, Routes } from "react-router-dom";
import Home from "./pages/Home";
import UploadBoth from "./pages/UploadBoth";
import UploadKey from "./pages/UploadKey";
import UploadSheet from "./pages/UploadSheet";
import Evaluate from "./pages/Evaluate";
import Results from "./pages/Results";

export default function App() {
  return (
    <div className="app-shell">
      <header>
        <h1>GRADE — Handwritten answer grading</h1>
        <nav>
          <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>
            Home
          </NavLink>
          <NavLink to="/upload" className={({ isActive }) => (isActive ? "active" : "")}>
            Upload (key + sheet)
          </NavLink>
          <NavLink to="/upload-key" className={({ isActive }) => (isActive ? "active" : "")}>
            Key only
          </NavLink>
          <NavLink to="/upload-sheet" className={({ isActive }) => (isActive ? "active" : "")}>
            Sheet only
          </NavLink>
          <NavLink to="/evaluate" className={({ isActive }) => (isActive ? "active" : "")}>
            Evaluate
          </NavLink>
          <NavLink to="/results" className={({ isActive }) => (isActive ? "active" : "")}>
            Results
          </NavLink>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/upload" element={<UploadBoth />} />
          <Route path="/upload-key" element={<UploadKey />} />
          <Route path="/upload-sheet" element={<UploadSheet />} />
          <Route path="/evaluate" element={<Evaluate />} />
          <Route path="/results" element={<Results />} />
        </Routes>
      </main>
    </div>
  );
}
