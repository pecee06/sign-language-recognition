const App = () => {
	return (
		<main className="flex flex-col items-center justify-center min-h-screen bg-gray-200">
			<button
				className="cursor-pointer border-none outline-none px-4 py-2 rounded bg-amber-200 text-green-900 font-bold"
				onClick={(e) => {
					e.preventDefault();
					fetch("/api/detect", { method: "GET" })
						.then((response) => {
							if (!response.ok) {
								throw new Error("Network response was not ok");
							}
							return response.json();
						})
						.then((data) => {
							console.log(data);
						})
						.catch((error) => {
							console.error(
								"There was a problem with the fetch operation:",
								error
							);
						});
				}}
			>
				Detect
			</button>
		</main>
	);
};

export default App;
