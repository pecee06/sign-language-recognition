const Button = ({ text, func = () => {}, style }) => {
	return (
		<button
			className={`cursor-pointer border-none outline-none px-4 py-2 rounded ${style}`}
			onClick={func}
		>
			{text}
		</button>
	);
};

export default Button;
