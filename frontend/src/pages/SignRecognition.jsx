import { Button, Screen } from "../components";
import { useRef, useState } from "react";
import textToSpeech from "../utils/textToSpeech.js";

const SignRecognition = () => {
	const screenRef = useRef(null);
	const [speaking, setSpeaking] = useState(false);

	return (
		<section className="flex flex-col items-center min-h-screen bg-violet-400 gap-[2vmax] p-10">
			<Button
				text="Recognize Sign"
				style="bg-white text-violet-500 hover:bg-violet-600 hover:text-white"
				func={() => {
					fetch("/api/recognize", { method: "GET" })
						.then((response) => {
							if (!response.ok) throw new Error("Network response was not ok");
							return response.json();
						})
						.then((data) => {
							if (screenRef.current) screenRef.current.value += data.word + " ";
						})
						.catch((error) => {
							console.error(
								"There was a problem with the fetch operation:",
								error
							);
						});
				}}
			/>
			<hr className="h-0.5 w-full bg-gray-600 outline-none border-none" />
			<Screen
				style="bg-white min-h-[40vh] w-full"
				ref={screenRef}
			/>
			<div className="flex gap-2 mt-4">
				<Button
					text={`${speaking ? "Stop" : "Speak"}`}
					style="bg-white text-violet-500 hover:bg-violet-600 hover:text-white"
					func={() => {
						if (screenRef.current) {
							const text = screenRef.current.value.trim();
							if (text) textToSpeech(text);
							else alert("There's nothing to read!");
						}
					}}
				/>
				<Button
					text="Clear"
					style="bg-white text-violet-500 hover:bg-violet-600 hover:text-white"
					func={() => {
						if (screenRef.current) screenRef.current.value = "";
					}}
				/>
			</div>
		</section>
	);
};

export default SignRecognition;
