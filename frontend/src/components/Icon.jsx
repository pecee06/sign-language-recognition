import { useState } from "react";

const Icon = ({ style }) => {
	const [showText, setShowText] = useState(false);

	return (
		<div
			onMouseEnter={() => setShowText(true)}
			onMouseLeave={() => setShowText(false)}
			className={`cursor-default ${style}`}
		>
			<p
				className={`text-gray-700 ${
					showText ? "block" : "hidden"
				} bg-white min-h-[62vh] rounded-lg p-4 flex justify-center items-center`}
			>
				It's essential to recognize that many individuals with hearing
				impairments face challenges due to societal misconceptions and lack of
				awareness. Terms like "deaf and dumb" are outdated and considered
				offensive; it's more appropriate to use "deaf" or "hard of hearing" when
				referring to individuals with hearing impairments.
				<br />
				Promoting inclusive education, accessible communication methods
				&#40;like sign language&#41;, and supportive technologies can
				significantly enhance the quality of life for the deaf and
				hard-of-hearing community.
			</p>
			<img
				src="/people-talking-in-sign-language.jpg"
				className={`object-cover rounded-lg ${showText ? "hidden" : "block"}`}
			/>
		</div>
	);
};

export default Icon;
