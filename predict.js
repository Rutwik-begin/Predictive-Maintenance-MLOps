
function sigmoid(x) {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }

  const ex = Math.exp(x);
  return ex / (1 + ex);
}

function evaluateTree(tree, features) {
  let node = 0;

  while (tree.children_left[node] !== -1) {
    const featureIndex = tree.feature[node];
    const threshold = tree.threshold[node];

    if (features[featureIndex] <= threshold) {
      node = tree.children_left[node];
    } else {
      node = tree.children_right[node];
    }
  }

  return tree.value[node];
}

function predictFailure(modelData, input) {
  const features = modelData.features.map((name) => {
    if (!(name in input)) {
      throw new Error(`Missing input feature: ${name}`);
    }

    const value = Number(input[name]);

    if (!Number.isFinite(value)) {
      throw new Error(`Invalid numeric value for: ${name}`);
    }

    return value;
  });

  if (features.length !== modelData.n_features) {
    throw new Error("Input feature count does not match the model.");
  }

  let rawScore = modelData.initial_score;

  for (const tree of modelData.trees) {
    rawScore += modelData.learning_rate *
      evaluateTree(tree, features);
  }

  const failureProbability = sigmoid(rawScore);

  return {
    prediction: failureProbability >= 0.5 ? 1 : 0,
    failureProbability: failureProbability,
    failurePercentage: failureProbability * 100,
  };
}

// Make the function available to a browser page.
window.predictFailure = predictFailure;
