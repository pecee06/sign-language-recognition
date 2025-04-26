import { Button, Title, Icon, Table } from "./components";
import { useNavigate } from "react-router";

const App = () => {
	const navigate = useNavigate();

	return (
		<main className="flex flex-col items-center min-h-screen bg-violet-400 gap-[2vmax]">
			<Title />
			<div className="flex gap-1 p-2">
				<Icon style="w-[35vmax]" />
				<Table
					data={[
						{
							Problem: "Communication Barriers",
							Explanation:
								"Daily conversations with hearing individuals become difficult because most people don't know sign language"
						},
						{
							Problem: "Limited Access to Services",
							Explanation:
								"At hospitals, government offices, and workplaces, lack of sign language interpreters can lead to misunderstandings or denial of essential services"
						},
						{
							Problem: "Social Isolation",
							Explanation:
								"Feeling left out of group conversations, events, or social gatherings leads to loneliness and mental health struggles"
						},
						{
							Problem: "Employment Difficulties",
							Explanation:
								"Bias and communication gaps can limit job opportunities, even for highly skilled deaf individuals"
						},
						{
							Problem: "Educational Challenges",
							Explanation:
								"Lack of inclusive education methods can prevent deaf students from fully accessing academic content"
						},
						{
							Problem: "Emergency Situations",
							Explanation:
								"In times of urgent need (e.g., natural disasters, medical emergencies), they might not be able to quickly communicate vital information"
						}
					]}
				/>
			</div>
			<Button
				text="Try it out!"
				style="bg-white text-violet-500 hover:bg-violet-600 hover:text-white transition-all duration-300"
				func={(e) => {
					e.preventDefault();
					navigate("/recognize");
				}}
			/>
		</main>
	);
};

export default App;
