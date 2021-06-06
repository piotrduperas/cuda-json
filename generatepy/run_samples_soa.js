const { exec } = require('child_process');
const fs = require("fs");


const execute = (command, callback) => {
	return new Promise((resolve) => {
		exec(command, (error, stdout, stderr) => { resolve(stdout); console.log(stderr)});
	});
};

const findTimeOf = (part, output) => {
	return output.split(`${part}: `)[1].split('ms')[0];
}

const samples = [];
const results = [];

for(let i = 1; i <= 9; i++){
	for(let j = 1; j <= 5; j++){
		samples.push(`sample_${i}${'0'.repeat(j)}.json`);
	}
}


(async () => {


	for(const sample of samples){
		const stats = fs.statSync(sample);
		const fileSizeInBytes = stats.size;

		const output = await execute(`../mainsoa ${sample}`);

		const reading = findTimeOf("Reading file", output);
		const cpu = findTimeOf("Cpu alogithm", output);
		const copying = findTimeOf("Copying file", output);
		const calculations = findTimeOf("Calculations", output);

		results.push({
			sample,
			fileSizeInBytes,
			reading,
			copying,
			calculations,
			cpu
		});
	}

	results.sort((a, b) => a.fileSizeInBytes > b.fileSizeInBytes ? 1 : -1);

	for(const result of results){
		console.log(Object.values(result).join('\t'));
	}

})();

