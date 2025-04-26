import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router";
import "./index.css";
import App from "./App.jsx";
import SignRecognition from "./pages/SignRecognition.jsx";

createRoot(document.getElementById("root")).render(
	<BrowserRouter>
		<Routes>
			<Route
				path="/"
				element={<App />}
			/>
			<Route
				path="/recognize"
				element={<SignRecognition />}
			/>
		</Routes>
	</BrowserRouter>
);
