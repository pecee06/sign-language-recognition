const Table = ({ data = [] }) => {
	// data is an array of objects
	let columns = Object.keys(data[0] || {});

	return (
		<table className="border-collapse border border-gray-300 text-left text-gray-700 min-h-[62vh]">
			<thead className="bg-gray-100">
				<tr>
					{columns.map((column) => (
						<th
							key={column}
							className="border border-gray-300 px-4 py-2 font-semibold"
						>
							{column}
						</th>
					))}
				</tr>
			</thead>
			<tbody>
				{data.map((row, index) => (
					<tr
						key={index}
						className="hover:bg-gray-50 transition-all duration-300"
					>
						{columns.map((column) => (
							<td
								key={column}
								className="border border-gray-300 px-4 py-2"
							>
								{row[column]}
							</td>
						))}
					</tr>
				))}
			</tbody>
		</table>
	);
};

export default Table;
