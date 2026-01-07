import { pipeline, env } from "@xenova/transformers";
import path from 'path';
import { pathToFileURL } from 'url';

env.localModelPath = path.resolve('.');
env.allowRemoteModels = false;
env.allowLocalModels = true;

const p = path.resolve('./onnx_output');
const modelPath = pathToFileURL(p).href;
console.log("⚡ Loading the custom safety AI...")

let classifier = await pipeline('text-classification', 'onnx_output', {
    quantized: false, // Full precision
    model_file_name: 'model'
});

const LABELS = [
    "SFW",
    "Controlled/Regulated Substances",
    "Copyright/Trademark/Plagiarism",
    "Criminal Planning/Confessions",
    "Fraud/Deception",
    "Guns and Illegal Weapons",
    "Harassment",
    "Hate/Identity Hate",
    "High Risk Gov Decision Making",
    "Illegal Activity",
    "Immoral/Unethical",
    "Malware",
    "Manipulation",
    "Needs Caution",
    "Other",
    "PII/Privacy",
    "Political/Misinformation/Conspiracy",
    "Profanity",
    "Sexual",
    "Sexual (minor)",
    "Suicide and Self Harm",
    "Threat",
    "Unauthorized Advice",
    "Violence"
];

const POLICY_WEIGHTS = {
    // Nuclear (immediate action)
    "Sexual (minor)": 10,
    "Suicide and Self Harm": 10,
    "Hate/Identity Hate": 6,
    "Violence": 6,
    // Severe (high risk)
    "Criminal Planning/Confessions": 4,
    "Sexual": 4,
    "Threat": 4,
    // Mild
    "Harassment": 2,
    "Profanity": 1,
    "Needs Caution": 1,
    "SFW": 0
};

async function assessRisk(text) {
    let raw_outputs = await classifier(text, { top_k: null});
    let maxRisk = 0;
    let primaryCategory = "SFW";

    for (let item of raw_outputs) {
        let id = parseInt(item.label.replace("LABEL_", ""));
        let realLabel = LABELS[id];

        let weight = POLICY_WEIGHTS[realLabel] || 1; 
        let riskScore = item.score * weight;

        // Track the highest risk found
        if (riskScore > maxRisk) {
            maxRisk = riskScore;
            primaryCategory = realLabel;
        }

        if (item.score > 0.05) {
             console.log(`   > Detected: ${realLabel} (${(item.score*100).toFixed(1)}%)`);
        }
    }

    // "Traffic Light"
    if (maxRisk >= 4.5) return { action: "AUTO_BAN", reason: primaryCategory, risk: maxRisk.toFixed(2) };
    else if (maxRisk >= 2.0) return { action: "HUMAN_REVIEW", reason: primaryCategory, risk: maxRisk.toFixed(2) };
    else return { action: "ALLOW", reason: primaryCategory, risk: maxRisk.toFixed(2) };
}

/* RUN TESTS */
console.log("✅ System Online. Running tests...\n");

const tests = [
    "I will kill everyone", /* Violence? */
    "You're stupid.", /* Should be SFW */
    "I want to steal the moon", /* Criminal? Human review most likely */
    "Hello friend, how are you?", /* SFW */
    "I am going to self harm" /* Self harm */
];

for (const t of tests) {
    const result = await assessRisk(t);
    console.log(`Input: "${t}"`);
    console.log(`Result: [${result.action}] Reason: ${result.reason} (Risk: ${result.risk})`);
    console.log("-".repeat(40));
}