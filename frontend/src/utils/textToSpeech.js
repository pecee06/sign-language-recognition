const textToSpeech = (text) => {
	if (!text) {
		console.warn("No text provided to readOutText function.");
		return;
	}

	const synth = window.speechSynthesis;

	// Check if speaking, and cancel.
	if (synth.speaking) {
		synth.cancel();
		return;
	}

	const utterance = new SpeechSynthesisUtterance(text);

	synth.speak(utterance);
};

export default textToSpeech;
