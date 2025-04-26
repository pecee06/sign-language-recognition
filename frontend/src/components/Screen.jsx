import { forwardRef } from "react";

const Screen = forwardRef(({ style }, ref) => {
	return (
		<textarea
			readOnly
			className={`w-full h-auto p-2 border rounded shadow-sm focus:outline-none focus:ring focus:ring-indigo-200 ${style}`}
			ref={ref}
		/>
	);
});

export default Screen;
