document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("fraud-form");
  const resultCard = document.getElementById("result-card");
  const titleEl = document.getElementById("result-title");
  const summaryEl = document.getElementById("result-summary");
  const probTwoEl = document.getElementById("prob-two-layer");
  const probBaseEl = document.getElementById("prob-baseline");
  const preScoreEl = document.getElementById("pre-score");
  const preFlagEl = document.getElementById("pre-flag");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const amount = parseFloat(document.getElementById("amount").value);
    const fromAccount = document.getElementById("from_account").value;
    const toAccount = document.getElementById("to_account").value;
    const template = document.getElementById("template").value;

    if (Number.isNaN(amount)) {
      alert("Please enter a valid numeric value for Amount.");
      return;
    }

    if (!fromAccount || !toAccount) {
      alert("Please enter both From and To account numbers.");
      return;
    }

    try {
      console.log("Sending prediction request...", { amount, fromAccount, toAccount, template });
      
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ amount, template, from_account: fromAccount, to_account: toAccount }),
      });

      console.log("Response status:", res.status, res.ok);
      
      const data = await res.json();
      console.log("Response data:", data);

      if (!res.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      const probTwo = data.prob_two_layer || data.probTwoLayer || 0;
      const probBase = data.prob_baseline || data.probBaseline || 0;
      const isFraud = data.prediction === 1;
      const preScore = data.pre_score;
      const preFlag = data.pre_flag;
      const trueClass = data.synthetic_true_class;
      const fromAcc = data.from_account || fromAccount;
      const toAcc = data.to_account || toAccount;
      const pre_message = data.pre_message || "";
      const post_message = data.post_message || "";

      console.log("Parsed data:", { probTwo, probBase, isFraud, preScore, preFlag, trueClass });

      // Title + summary — prefer backend-provided `payment_display` if present
      const paymentDisplay = data.payment_display || (isFraud ? "⚠️ Fraud detected — payment stopped" : "✅ No fraud detected — payment completed");
      titleEl.textContent = paymentDisplay;

      // Short summary line (payment display already shown in title)
      summaryEl.textContent = `Two-layer fraud probability: ${(probTwo * 100).toFixed(2)} • Baseline: ${(probBase * 100).toFixed(2)}`;

      // Detailed metrics — put each into its own line/slot
      probTwoEl.textContent = (probTwo * 100).toFixed(2) + "%";
      probBaseEl.textContent = (probBase * 100).toFixed(2) + "%";
      preScoreEl.textContent = preScore.toFixed(4);
      preFlagEl.textContent = preFlag === 1 ? "Suspicious" : "Normal";

      // From/To and messages
      document.getElementById("from-account").textContent = fromAcc;
      document.getElementById("to-account").textContent = toAcc;
      document.getElementById("pre-message").textContent = pre_message;
      document.getElementById("post-message").textContent = post_message;

      resultCard.classList.remove("hidden");
      console.log("Results displayed successfully");
    } catch (err) {
      console.error("Error:", err);
      alert("Error while predicting: " + err.message);
    }
  });
});
