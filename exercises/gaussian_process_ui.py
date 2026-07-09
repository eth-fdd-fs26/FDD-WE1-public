from IPython.display import HTML

def show_introduction():
    html_content = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
                max-width: 900px; margin: 20px auto; padding: 30px; background-color: #f8fafc; 
                border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 20px rgba(0,0,0,0.03); color: #334155;">
        <div style="text-align: center; margin-bottom: 25px;">
            <span style="background: #e0f2fe; color: #0369a1; padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
                🌍 Introduction
            </span>
            <h2 style="color: #1e293b; margin-top: 10px; margin-bottom: 0; font-size: 1.6rem; font-weight: 800;">Earth's Breathing Cycle: Mauna Loa CO2</h2>
        </div>
        <p style="font-size: 0.95rem; line-height: 1.6; margin-bottom: 20px;">
            You have just been handed one of the most famous datasets in atmospheric science: the monthly carbon dioxide measurements taken at the Mauna Loa Observatory in Hawaii. Your objective is to build a model capable of <strong>extrapolating</strong> the CO2 concentration decades into the future.
        </p>
        <p style="font-size: 0.95rem; line-height: 1.6; margin-bottom: 25px;">
            Standard linear regressions or simple neural networks often fail here. You will use a <strong>Gaussian Process (GP)</strong> to construct a non-parametric model that learns both the long-term human impact and the Earth's natural seasonal cycles by composing different Covariance Kernels.
        </p>
        <p style="font-size: 0.95rem; line-height: 1.6; margin-bottom: 25px;">
            The dataset your are working with contains 4 columns: year, month, decimal date, and the CO2 concentration in parts per million (ppm). The decimal date is a continuous representation of time, which is useful for modeling. The CO2 concentration is the target variable you will be predicting based on the decimal date.
        </p>
    </div>
    """
    return HTML(html_content)

def show_composite_strategy():
    """Generates an HTML explanation of composite kernels including the Noise Component."""
    html_content = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
                max-width: 900px; margin: 20px auto; padding: 30px; background-color: #f8fafc; 
                border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 20px rgba(0,0,0,0.03);
                color: #334155;">
        
        <div style="text-align: center; margin-bottom: 25px;">
            <span style="background: #f3e8ff; color: #6b21a8; padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
                🧩 Kernel Algebra
            </span>
            <h2 style="color: #1e293b; margin-top: 10px; margin-bottom: 0; font-size: 1.6rem; font-weight: 800; letter-spacing: -0.025em;">
                Building the Climate Kernel
            </h2>
        </div>
        
        <p style="font-size: 0.95rem; line-height: 1.6; margin-top: 0; margin-bottom: 25px; color: #475569; text-align: left;">
            As you just saw, a single stationary kernel fails to extrapolate. The magic of Gaussian Processes lies in <strong>kernel composition</strong>. By adding and multiplying kernels, we can encode complex structural priors. We will build our composite kernel using three components:
        </p>
        
        <div style="background: white; padding: 20px; border-radius: 12px; border-top: 5px solid #3b82f6; box-shadow: 0 2px 8px rgba(0,0,0,0.01); margin-bottom: 20px;">
            <h5 style="margin: 0 0 10px 0; color: #1e293b; font-size: 1.05rem; font-weight: 700; display: flex; justify-content: space-between;">
                <span>1. The Long-Term Trend</span>
                <math display="inline"><msub><mi>K</mi><mi>Trend</mi></msub></math>
            </h5>
            <p style="margin: 0; color: #64748b; font-size: 0.9rem; line-height: 1.5;">
                An RBF kernel with a <em>large</em> length-scale to capture the smooth, decades-long rise in CO2. It ignores monthly wiggles and focuses on the macroeconomic trajectory. For flexibility, we multiply this by a ConstantKernel to allow the overall amplitude (vertical scale)of the trend to be learned from the data.
            </p>
        </div>

        <div style="background: white; padding: 20px; border-radius: 12px; border-top: 5px solid #10b981; box-shadow: 0 2px 8px rgba(0,0,0,0.01); margin-bottom: 20px;">
            <h5 style="margin: 0 0 10px 0; color: #1e293b; font-size: 1.05rem; font-weight: 700; display: flex; justify-content: space-between;">
                <span>2. The Seasonal Cycle</span>
                <math display="inline"><msub><mi>K</mi><mi>Periodic</mi></msub></math>
            </h5>
            <p style="margin: 0; color: #64748b; font-size: 0.9rem; line-height: 1.5;">
                We use the ExpSineSquared kernel, which is a periodic kernel captures the seasonal oscillation of the data.
                This kernel has two hyperparameters: the <code>length_scale</code> controls how smooth the periodic oscillation is, and the <code>periodicity</code> controls how long it takes for the cycle to repeat. Intuitively, the periodicity should be close to 1 (year) in our case.
                We also multiply this by a ConstantKernel to learn the vertical scaling from data.
            </p>
        </div>

        <div style="background: white; padding: 20px; border-radius: 12px; border-top: 5px solid #f59e0b; box-shadow: 0 2px 8px rgba(0,0,0,0.01); margin-bottom: 20px;">
            <h5 style="margin: 0 0 10px 0; color: #1e293b; font-size: 1.05rem; font-weight: 700; display: flex; justify-content: space-between;">
                <span>3. The Noise Component</span>
                <math display="inline"><msub><mi>K</mi><mi>Noise</mi></msub></math>
            </h5>
            <p style="margin: 0; color: #64748b; font-size: 0.9rem; line-height: 1.5;">
                This is a WhiteKernel that accounts for irregularities, sensor fluctuations, and localized disruptions in monthly measurements. Instead of forcing the main GP curves to rigidly twist through every single training point, the noise kernel acts as a shock absorber to prevent overfitting.
            </p>
        </div>
        
        <div style="margin-top: 25px; background: #f1f5f9; padding: 15px; border-radius: 10px; color: #1e293b; text-align: center; width: 100%;">
            <math display="block">
                <msub><mi>K</mi></msub>
                <mo>=</mo>
                <msub><mi>K</mi><mi>Trend</mi></msub>
                <mo>+</mo>
                <msub><mi>K</mi><mi>Periodic</mi></msub>
                <mo>+</mo>
                <msub><mi>K</mi><mi>Noise</mi></msub>
            </math>
        </div>
    </div>
    """
    return HTML(html_content)


def show_lml_explanation():
    html_content = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                max-width: 900px; margin: 20px auto; padding: 30px; background-color: #f8fafc; 
                border-radius: 16px; border: 1px solid #e2e8f0; color: #334155;">
        
        <div style="text-align: center; margin-bottom: 25px;">
            <span style="background: #f3e8ff; color: #6b21a8; padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
                🧮 Hyperparameter Selection
            </span>
            <h2 style="color: #1e293b; margin-top: 10px; margin-bottom: 0; font-size: 1.6rem; font-weight: 800; letter-spacing: -0.025em;">
                Maximizing the Log Marginal Likelihood (LML)
            </h2>
        </div>
        
        <p style="font-size: 0.95rem; line-height: 1.6;">
            Manually tuning kernel sliders is great for intuition, but how can we find the "optimal" kernel hyperparameters systemically? In Gaussian Processes, the standard approach is to maximize the <strong>Log Marginal Likelihood (LML)</strong>. This is exactly what happens in Scikit-learn when you call <code>GaussianProcessRegressor.fit()</code>!
        </p>
        
        <h4 style="color: #475569; margin-bottom: 8px; font-size: 1.1rem;">The Core Intuition</h4>
        <p style="font-size: 0.95rem; line-height: 1.6; margin-top: 0;">
            In standard (frequentist) machine learning, we score a model based on how well <em>one specific function</em> fits the data. This is often done by calculating the likelihood: the probability of observing the data if the function were the actual underlying mechanism.
        </p>

        <p style="font-size: 0.95rem; line-height: 1.6; margin-top: 0;">
            In Gaussian Processes, we have an infinite number of possible functions, and the posterior tells us how likely each function is given the observed data.
            This allows us to compute the average of the likelihoods of all possible functions, weighted by how likely each function is. This average is called the <strong>marginal likelihood</strong>.

            We aren't asking how well a single curve fit the data. We are asking how well our <em>belief about the underlying mechanism</em> (i.e. the posterior) explains the data on average.
        </p>

        <p style="font-size: 0.95rem; line-height: 1.6; margin-top: 0;">
            By taking the logarithm of this marginal likelihood, we get the <strong>Log Marginal Likelihood (LML)</strong>, which is easier to work with numerically and mathematically. Maximizing the LML allows us to find the kernel hyperparameters that make our belief about the underlying function most consistent with the observed data.
        </p>

        <div style="font-size: 0.95rem; line-height: 1.6; background: #eff6ff; padding: 20px; border-radius: 8px; border-left: 4px solid #1d4ed8; margin: 20px 0;">
            <h4 style="color: #1e3a8a; margin-top: 0; margin-bottom: 10px; font-size: 1.05rem;">The Occam's Razor Trade-off</h4>
            <p style="margin-top: 0;">
                Maximizing the LML is powerful because it naturally encodes <strong>Occam's Razor</strong>—the philosophical principle that when presented with competing hypotheses, the simplest explanation is usually the best one. The LML achieves this by balancing two competing forces:
            </p>
            <ul style="margin-bottom: 0; padding-left: 20px;">
                <li style="margin-bottom: 10px;">
                    <strong>Data Fit:</strong> How closely the generated curves pass through the actual training points. A model with higher capacity (e.g., a very small lengthscale) can fit the training data extremely well.
                </li>
                <li>
                    <strong>Complexity Penalty:</strong> How "wiggly" or complex the model is. Higher model complexity leads to a larger penalty, which discourages overfitting.
                </li>
            </ul>
        </div>
        
        <p style="font-size: 0.95rem; line-height: 1.6;">
            By maximizing the LML, the GP automatically finds the sweet spot: the simplest possible curve that explains your data without overfitting. Click the <strong>"Optimize LML"</strong> button below to select the best hyperparameters for your composite kernel automatically.
        </p>
    </div>
    """
    return HTML(html_content)

def rbf_quiz_1():
    html_content = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                max-width: 900px; margin: 25px auto; padding: 25px; background-color: #f8fafc; 
                border-radius: 16px; border: 1px solid #e2e8f0; color: #334155;">
        <span style="background: #e0f2fe; color: #0369a1; padding: 5px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700;">🧠 Knowledge Check: RBF Properties</span>
        <h4 style="color: #1e293b; margin-top: 10px;">Which of the following statements correctly describe the behavior of our RBF Gaussian Process model? (Select all that apply)</h4>
        
        <div style="display: flex; flex-direction: column; gap: 12px; margin-bottom: 20px; font-size: 0.95rem; line-height: 1.5;">
            <label style="display: flex; align-items: flex-start; gap: 8px; cursor: pointer;">
                <input type="checkbox" name="rbf_quiz_opts" value="1" style="margin-top: 4px;"> 
                <span>1. Uncertainty of our GP prediction widens as we go further into the future.</span>
            </label>
            <label style="display: flex; align-items: flex-start; gap: 8px; cursor: pointer;">
                <input type="checkbox" name="rbf_quiz_opts" value="2" style="margin-top: 4px;"> 
                <span>2. Smaller RBF length scale leads to increased model complexity, allowing the model to fit local data more accurately at the expense of longer-term trends.</span>
            </label>
            <label style="display: flex; align-items: flex-start; gap: 8px; cursor: pointer;">
                <input type="checkbox" name="rbf_quiz_opts" value="3" style="margin-top: 4px;"> 
                <span>3. As we go further into the future, covariance with the training data drops to zero. This gives predictions close to the prior mean with variance close to the Constant Kernel variance.</span>
            </label>
        </div>
        
        <button onclick="checkRbfQuizAnswers()" style="background: #2563eb; color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 600; cursor: pointer;">Submit Answers</button>
        <div id="rbf_quiz_feedback" style="margin-top: 15px; display: none; padding: 15px; border-radius: 10px; line-height: 1.5;"></div>
    </div>

    <script>
    function checkRbfQuizAnswers() {
        const checkedBoxes = document.querySelectorAll('input[name="rbf_quiz_opts"]:checked');
        const feedback = document.getElementById('rbf_quiz_feedback');
        feedback.style.display = "block";
        
        const selectedValues = Array.from(checkedBoxes).map(cb => cb.value);
        
        // All 3 statements are mathematically true and correct properties of stationary RBF kernels!
        if (selectedValues.includes("1") && selectedValues.includes("2") && selectedValues.includes("3") && selectedValues.length === 3) {
            feedback.style.background = "#dcfce7"; 
            feedback.style.color = "#166534";
            feedback.innerHTML = `
                <strong>🎉 Brilliant! All three statements are true:</strong><br>
                <ul>
                    <li><strong>Statement 1 & 3 (Stationarity & Prior Reversion):</strong> Because the pure RBF kernel is stationary, as your test point moves away from historical observations, the covariance decays exponentially to 0. The GP reverts entirely to its prior mean, and the predictive variance maximizes to match your Constant Kernel variance, creating widening bands that eventually flatline.</li>
                    <li><strong>Statement 2 (Lengthscale & Complexity):</strong> Shorter lengthscales allow the covariance matrix to decorrelate rapidly over short intervals, causing highly flexible, local wiggliness instead of identifying long-term growth trends.</li>
                </ul>
            `;
        } else {
            feedback.style.background = "#fffbeb"; 
            feedback.style.color = "#92400e";
            feedback.innerHTML = "<strong>⚠️ Not quite.</strong> Re-read your choices. Every single one of these statements accurately defines a fundamental rule of how an RBF Gaussian Process behaves when extrapolating data. Check all options that apply!";
        }
    }
    </script>
    """
    return HTML(html_content)

def rbf_quiz_2():
    html_content = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                max-width: 900px; margin: 25px auto; padding: 25px; background-color: #f8fafc; 
                border-radius: 16px; border: 1px solid #e2e8f0; color: #334155;">
        <span style="background: #fee2e2; color: #991b1b; padding: 5px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700;">🧠 Knowledge Check: Model Suitability</span>
        <h4 style="color: #1e293b; margin-top: 10px; margin-bottom: 5px;">Why is a standalone Radial Basis Function (RBF) kernel mathematically unsuitable for modeling and long-term extrapolation of the Mauna Loa CO2 concentration data?</h4>
        <p style="font-size: 0.85rem; color: #64748b; margin-top: 0; margin-bottom: 20px;">(Select all that apply)</p>
        
        <div style="display: flex; flex-direction: column; gap: 14px; margin-bottom: 20px; font-size: 0.95rem; line-height: 1.5;">
            <label style="display: flex; align-items: flex-start; gap: 10px; cursor: pointer;">
                <input type="checkbox" name="rbf_suitability_opts" value="1" style="margin-top: 4px;"> 
                <span><strong>Option 1:</strong> The true underlying carbon trend is non-stationary and exhibits a long-term upward trajectory, whereas the RBF kernel is inherently stationary.</span>
            </label>
            <label style="display: flex; align-items: flex-start; gap: 10px; cursor: pointer;">
                <input type="checkbox" name="rbf_suitability_opts" value="2" style="margin-top: 4px;"> 
                <span><strong>Option 2:</strong> The RBF kernel calculates smooth, continuous distance decay but possesses no built-in cyclic or periodic mechanics to model the Earth's annual breathing seasonality.</span>
            </label>
            <label style="display: flex; align-items: flex-start; gap: 10px; cursor: pointer;">
                <input type="checkbox" name="rbf_suitability_opts" value="3" style="margin-top: 4px;"> 
                <span><strong>Option 3:</strong> The RBF kernel can successfully capture both the upward trend and seasonality if we simply set the <code>length_scale</code> hyperparameter to a sufficient small value.</span>
            </label>
        </div>
        
        <button onclick="checkRbfSuitabilityAnswers()" style="background: #2563eb; color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 600; cursor: pointer;">Submit Answers</button>
        <div id="rbf_suitability_feedback" style="margin-top: 15px; display: none; padding: 18px; border-radius: 12px; line-height: 1.6; font-size: 0.95rem;"></div>
    </div>

    <script>
    function checkRbfSuitabilityAnswers() {
        const checkedBoxes = document.querySelectorAll('input[name="rbf_suitability_opts"]:checked');
        const feedback = document.getElementById('rbf_suitability_feedback');
        feedback.style.display = "block";
        
        const selectedValues = Array.from(checkedBoxes).map(cb => cb.value);
        
        // Options 1 and 2 are the correct choices. Option 3 is incorrect.
        const correctSelection = selectedValues.includes("1") && selectedValues.includes("2") && !selectedValues.includes("3") && selectedValues.length === 2;
        
        if (correctSelection) {
            feedback.style.background = "#dcfce7"; 
            feedback.style.color = "#166534";
            feedback.innerHTML = `
                <strong style="font-size: 1.05rem;">🎉 Spot on! Options 1 and 2 are correct.</strong><br><br>
                <strong>Why Option 1 is true:</strong> Because the pure RBF kernel is stationary, its prior variance is constant across time. When generalizing into the future, the model lacks structural mechanics to continue a monotonic trend upwards, forcing the mean prediction to slide back down towards zero (or the global data mean) while the bounds max out to full prior uncertainty.<br><br>
                <strong>Why Option 2 is true:</strong> RBF assumes that data correlation relies purely on proximity. It has no awareness that a data point today is intrinsically highly correlated with a data point exactly 12 months later. Without adding an kernel with a periodic nature, it cannot track seasonality.<br><br>
                <strong>Why Option 3 is false:</strong> Dropping the lengthscale to a very small value causes the covariance between neighboring points to vanish rapidly. Instead of fixing the trend, this forces the GP to violently overfit local points into individual jagged spikes which does not extrapolate beyond the immediate neighborhood.
            `;
        } else {
            feedback.style.background = "#fffbeb"; 
            feedback.style.color = "#92400e";
            feedback.innerHTML = `
                <strong>⚠️ Not quite there yet.</strong><br>
                Remember, think about the physical features of the Mauna Loa Keeling curve (constant growth + cyclical summer/winter respiration cycles) and match them against the algebraic limitations of a basic stationary distance-decay kernel. Try checking exactly which options explain why it fails to capture those structural trends!
            `;
        }
    }
    </script>
    """
    return HTML(html_content)



def hyperparameter_quiz():
    html_content = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                max-width: 900px; margin: 25px auto; padding: 25px; background-color: #f8fafc; border-radius: 16px; border: 1px solid #e2e8f0; color: #334155;">
        <span style="background: #ede9fe; color: #5b21b6; padding: 5px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700;">🧠 Parameter Tradeoffs Check</span>
        <h4 style="color: #1e293b; margin-top: 10px;">Based on your experiments in the Composite Dashboard, which statements are true?</h4>
        <div style="display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px;">
            <label><input type="checkbox" name="hp_quiz" value="1"> Decreasing the Trend Lengthscale makes the long-term curve more jagged and prone to overfitting local data clumps.</label>
            <label><input type="checkbox" name="hp_quiz" value="2"> Setting a high Noise Variance causes the GP confidence intervals to widen, acknowledging that the exact data points shouldn't be trusted perfectly.</label>
        </div>
        <button onclick="checkHpQuiz()" style="background: #2563eb; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer;">Submit Answers</button>
        <div id="hp_feedback" style="margin-top: 15px; display: none; padding: 12px; border-radius: 10px;"></div>
    </div>
    <script>
    function checkHpQuiz() {
        const boxes = document.querySelectorAll('input[name="hp_quiz"]:checked');
        const feedback = document.getElementById('hp_feedback');
        feedback.style.display = "block";
        const vals = Array.from(boxes).map(b => b.value);
        if (vals.includes("1") && vals.includes("2")) {
            feedback.style.background = "#dcfce7"; feedback.style.color = "#166534";
            feedback.innerHTML = "<strong>🎉 Excellent!</strong> Both observations are correct.";
        } else {
            feedback.style.background = "#fffbeb"; feedback.style.color = "#92400e";
            feedback.innerHTML = "<strong>⚠️ Not quite.</strong> Ensure both correct attributes are checked.";
        }
    }
    </script>
    """
    return HTML(html_content)


def show_interpretation():
    html_content = """
    <div style="
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        max-width: 850px;
        margin: 20px auto;
        padding: 26px;
        background-color: #f8fafc;
        border-radius: 16px;
        box-shadow: 0 4px 24px rgba(15, 23, 42, 0.04);
        color: #1e293b;
        line-height: 1.6;
    ">
        <div style="border-bottom: 2px solid #e2e8f0; padding-bottom: 14px; margin-bottom: 22px;">
            <h2 style="color: #0f172a; margin: 0 0 6px 0; font-size: 1.65rem; font-weight: 800; letter-spacing: -0.025em;">
                🔍 Gaussian Process Architecture Interpretation
            </h2>
        </div>

        <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; margin-bottom: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.01);">
            <h4 style="color: #0f172a; margin: 0 0 8px 0; font-size: 1.1rem; font-weight: 700;">📊 Epistemic Uncertainty vs. Hand-Tuning</h4>
            <p style="margin: 0; font-size: 0.95rem; color: #334155;">
                Because the LML optimizer chose parameters that fit the training data tightly, its confidence interval is very narrow where data is present. As it leaves the data zone, the confidence envelope widens significantly. This is a very honest representation of epistemic uncertainty: the model admitting it doesn't know what happens next.
            </p>
        </div>

        <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; margin-bottom: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.01);">
            <h4 style="color: #0f172a; margin: 0 0 8px 0; font-size: 1.1rem; font-weight: 700;">🛡️ Why "Reversing to the Mean" Is Not Completely Wrong</h4>
            <p style="margin: 0; font-size: 0.95rem; color: #334155;">
                In addition, the model maximizing the log-marginal likelihood does not have a large lengthscale as you might hope, which makes the predictions drop off to the mean in the future. This is not inherently wrong: If a sudden massive global policy change caused CO₂ emissions to drop to zero tomorrow, your manual model might confidently over-predict, while the LML model's wide bounds would still contain the true reality.
            </p>
        </div>

        <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; margin-bottom: 22px; box-shadow: 0 2px 6px rgba(0,0,0,0.01); border-left: 4px solid #3b82f6;">
            <h4 style="color: #1d4ed8; margin: 0 0 8px 0; font-size: 1.1rem; font-weight: 700;">⚙️ Modeling Choices vs Data & Optimization</h4>
            <p style="margin: 0; font-size: 0.95rem; color: #334155;">
                This exercise demonstrates that data and optimization do not substitute for proper kernel design. If you know for a fact that CO₂ will continue to rise, the structural solution is to replace the trend kernel with a non-stationary component, for example adding another a linear or polynomial kernel to capture the upwards trend on top of the RBF kernel.
            </p>
        </div>

        <div style="background-color: #f1f5f9; border-radius: 8px; padding: 12px 16px; display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.2rem;">📖</span>
            <p style="margin: 0; font-size: 0.9rem; color: #475569; font-weight: 500;">
                You can find a more sophisticated modelling choice in <strong>Chapter 4, Gaussian Processes for Machine Learning</strong> by Rasmussen & Williams.
            </p>
        </div>
    </div>
    """
    return HTML(html_content)

# ==================== DATA & VISUALIZATIONS ========================
# Initialize the interactive plot
import pandas as pd
import numpy as np
import calendar
from datetime import datetime, timedelta
from itertools import product
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, ExpSineSquared, WhiteKernel, ConstantKernel
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output


# =================== GP EXPLANATION AND VISUALIZATION ========================

def show_gp_explanation():
    """
    Renders a beautifully polished HTML explanation of the Gaussian Process framework
    designed to integrate seamlessly with standard Jupyter/Colab notebook dark/light themes.
    """
    html_content = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
                max-width: 950px; margin: 20px auto; padding: 35px; background-color: #f8fafc; 
                border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 25px rgba(0,0,0,0.02); color: #334155;">
        
        <div style="text-align: center; margin-bottom: 30px;">
            <span style="background: #f3e8ff; color: #6b21a8; padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
                🧠 Foundations
            </span>
            <h2 style="color: #1e293b; margin-top: 10px; margin-bottom: 12px; font-size: 1.75rem; font-weight: 800; letter-spacing: -0.02em;">
                The Mathematical Foundations of Gaussian Processes
            </h2>
            <div style="width: 50px; height: 4px; background: #8b5cf6; margin: 0 auto; border-radius: 2px;"></div>
        </div>

        <div style="background-color: #ffffff; border-left: 5px solid #8b5cf6; border-radius: 8px; padding: 20px; margin-bottom: 30px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
            <p style="font-size: 1rem; line-height: 1.6; margin: 0; color: #1e293b;">
                Formally speaking, a <strong>Gaussian Process (GP)</strong> is a (possibly infinite) collection of Gaussian random variables in which any finite subcollection is multivariate Gaussian. This behavior is entirely defined by two core objects:
            </p>
            <ul style="margin-top: 12px; margin-bottom: 0; padding-left: 20px; line-height: 1.6; font-size: 0.95rem;">
                <li style="margin-bottom: 6px;"><strong style="color: #6b21a8;">Mean Function:</strong> This tells you the expected value (i.e. mean) of the Gaussian distribution at each point <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>x</mi></math>.</li>
                <li><strong style="color: #6b21a8;">Kernel Function <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>K</mi><mo>(</mo><msub><mi>x</mi><mn>1</mn></msub><mo>,</mo><msub><mi>x</mi><mn>2</mn></msub><mo>)</mo></math>:</strong> This tells you the <em>covariance</em> (spatial or temporal alignment) between any pair of points <math xmlns="http://www.w3.org/1998/Math/MathML"><mo>(</mo><msub><mi>x</mi><mn>1</mn></msub><mo>,</mo><msub><mi>x</mi><mn>2</mn></msub><mo>)</mo></math>.</li>
            </ul>
        </div>

        <p style="font-size: 1rem; line-height: 1.6; margin-bottom: 25px;">
            To make things more concrete, we explain this concept in the context of our current task of <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mtext>CO</mtext><mn>2</mn></msub></math> concentration modeling: At any given continuous timestamp <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>x</mi></math>, we want to model the true carbon concentration curve <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>f</mi><mo>(</mo><mi>x</mi><mo>)</mo></math>.
        </p>

        <div style="display: grid; grid-template-columns: 1fr; gap: 25px;">
            
            <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.01);">
                <h3 style="color: #0f172a; margin-top: 0; margin-bottom: 10px; font-size: 1.2rem; font-weight: 700; display: flex; align-items: center; gap: 8px;">
                    📊 1. Functions as Infinite Collections of Variables
                </h3>
                <p style="font-size: 0.95rem; line-height: 1.6; margin: 0; color: #475569;">
                    Instead of thinking of a function <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>f</mi><mo>(</mo><mi>x</mi><mo>)</mo></math> as a fixed algebraic formula (like <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>y</mi><mo>=</mo><mi>m</mi><mi>x</mi><mo>+</mo><mi>b</mi></math>), a Gaussian Process provides a <strong>probabilistic</strong> framework over the possible choices of functions. 
                </p>
                <p style="font-size: 0.95rem; line-height: 1.6; margin-top: 10px; margin-bottom: 0; color: #475569;">
                    If we pluck a single isolated point in time <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>x</mi></math>, our model maps our current epistemic uncertainty as a single <strong>1D Gaussian Distribution (a classic bell curve)</strong>. In this worldview, <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>f</mi><mo>(</mo><mi>x</mi><mo>)</mo></math> is no longer a static point value; it is a normal distribution.
                </p>
                <p style="font-size: 0.95rem; line-height: 1.6; margin-top: 10px; margin-bottom: 0; color: #475569;">
                    When we group every single infinite continuous timestamp across our timeline simultaneously, we get the complete Gaussian Process. You can picture this visually as a dense, continuous sequence of normal distribution <strong>"slices" stacked side-by-side</strong> along the time axis. When we sample a single realization function from our model, we drop an elastic thread through these stacked slices—randomly catching one height coordinate from each bell curve.
                </p>
                <p style="font-size: 0.95rem; line-height: 1.6; margin-top: 10px; margin-bottom: 0; color: #475569;">
                    In other words, a Gaussian Process gives <strong>a probability distribution over functions</strong>.
                </p>
            </div>
            <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.01);">
                <h3 style="color: #0f172a; margin-top: 0; margin-bottom: 10px; font-size: 1.2rem; font-weight: 700; display: flex; align-items: center; gap: 8px;">
                    🔗 2. How the Kernel Controls the Slices (The Covariance Link)
                </h3>
                <p style="font-size: 0.95rem; line-height: 1.6; margin: 0; color: #475569;">
                    If every slice along the continuous timeline were completely isolated and independent, generating a sample curve from our prior would look like static noise—a chaotic, jagged explosion where the atmospheric value in January tells you absolutely nothing about the value in February.
                </p>
                <p style="font-size: 0.95rem; line-height: 1.6; margin-top: 10px; margin-bottom: 0; color: #475569;">
                    The kernel function <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>K</mi></math> acts as the geometric <strong>glue holding these slices together</strong>. It formulates exactly how any two arbitrary slices pull on one another:
                </p>
                
                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center; margin: 15px 0; font-size: 1.05rem;">
                    <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
                        <mtext>Cov</mtext>
                        <mo>(</mo>
                        <mi>f</mi>
                        <mo>(</mo>
                        <msub><mi>x</mi><mn>1</mn></msub><mo>)</mo>
                        <mo>,</mo>
                        <mi>f</mi>
                        <mo>(</mo>
                        <msub><mi>x</mi><mn>2</mn></msub><mo>)</mo>
                        <mo>)</mo>
                        <mo>=</mo>
                        <mi>K</mi>
                        <mo>(</mo>
                        <msub><mi>x</mi><mn>1</mn></msub><mo>,</mo>
                        <msub><mi>x</mi><mn>2</mn></msub><mo>)</mo>
                    </math>
                </div>

                <p style="font-size: 0.95rem; line-height: 1.6; margin-top: 10px; margin-bottom: 0; color: #475569;">
                    If <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>x</mi><mn>1</mn></msub></math> and <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>x</mi><mn>2</mn></msub></math> are close neighbors (e.g., just days or weeks apart), it is physically reasonable that their carbon levels remain highly correlated. If a wave spikes upward at <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>x</mi><mn>1</mn></msub></math>, the neighboring slice at <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>x</mi><mn>2</mn></msub></math> is mathematically dragged upward with it. 
                </p>
                <p style="font-size: 0.95rem; line-height: 1.6; margin-top: 10px; margin-bottom: 0; color: #475569;">
                    A fundamental, smooth choice to capture this phenomenon is the ubiquitous <strong>Radial Basis Function (RBF) Kernel</strong>:
                </p>

                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center; margin: 15px 0; font-size: 1.05rem;">
                    <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
                        <msub><mi>K</mi><mtext>RBF</mtext></msub>
                        <mo>(</mo>
                        <msub><mi>x</mi><mn>1</mn></msub><mo>,</mo>
                        <msub><mi>x</mi><mn>2</mn></msub><mo>)</mo>
                        <mo>=</mo>
                        <mi>exp</mi>
                        <mrow>
                            <mo>(</mo>
                            <mo>-</mo>
                            <mfrac>
                                <msup>
                                    <mrow>
                                        <mo>(</mo>
                                        <msub><mi>x</mi><mn>1</mn></msub>
                                        <mo>-</mo>
                                        <msub><mi>x</mi><mn>2</mn></msub>
                                        <mo>)</mo>
                                    </mrow>
                                    <mn>2</mn>
                                </msup>
                                <mrow>
                                    <mn>2</mn>
                                    <msup><mi>ℓ</mi><mn>2</mn></msup>
                                </mrow>
                            </mfrac>
                            <mo>)</mo>
                        </mrow>
                    </math>
                </div>

                <p style="font-size: 0.95rem; line-height: 1.6; margin-top: 10px; margin-bottom: 0; color: #475569;">
                    Here, <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>ℓ</mi></math> is the <strong>lengthscale</strong> hyperparameter controlling the smoothness of the GP realizations. A larger lengthscale values create long-range dependencies and force distant slices to maintain high correlations, which produces more gradually varying curves.
                </p>
            </div>
        </div>

        <div style="margin-top: 30px; padding: 15px; border-radius: 8px; background-color: #eff6ff; border: 1px solid #bfdbfe; font-size: 0.95rem; line-height: 1.5; color: #1e40af; font-style: italic; display: flex; align-items: center; gap: 10px;">
            🚀 <span>In the following cells, you will explore this slicing perspective and build your first scikit-learn Gaussian Process pipeline!</span>
        </div>
        
    </div>
    """
    return HTML(html_content)

def show_gp_anatomy(gp_model, X_min, X_max):
    """
    Renders an interactive 3D/2D visualization of a fitted sklearn GaussianProcessRegressor.
    
    Parameters:
    -----------
    gp_model (GaussianProcessRegressor): A scikit-learn GP model.
    X_min (float): The minimum value of the input domain.
    X_max (float): The maximum value of the input domain.
    """
    # 1. Setup our continuous domain representing the timeline
    X_line = np.linspace(X_min, X_max, int(X_max - X_min) * 10)
    mu_all = gp_model.predict(X_line[:, None], return_cov=False)
    

    # Create an Output widget container to hold the Plotly figure cleanly in notebooks
    colab_output = widgets.Output()

    # 2. Setup standard sliders for slicing through time indices
    x1_init = X_min + (X_max - X_min) * 0.25
    x2_init = X_min + (X_max - X_min) * 0.75
    slider_x1 = widgets.FloatSlider(value=x1_init, min=X_min, max=X_max, step=0.1, description='X1:', layout=widgets.Layout(width='400px'))
    slider_x2 = widgets.FloatSlider(value=x2_init, min=X_min, max=X_max, step=0.1, description='X2:', layout=widgets.Layout(width='400px'))

    def generate_gp_anatomy_plot(change=None):
        x1 = slider_x1.value
        x2 = slider_x2.value
        
        # Predict the GP mean and covariance for the two selected slices
        mu_, cov_ = gp_model.predict(np.array([[x1], [x2]]), return_cov=True)
        var_x1, var_x2, cov_x1_x2 = cov_[0, 0], cov_[1, 1], cov_[0, 1]
        mu_x1, mu_x2 = mu_[0], mu_[1]

        # --- Math: Reconstruct the 2D Joint Posterior Ellipse (Right Panel) ---
        theta = np.linspace(0, 2*np.pi, 100)
        ellipse_circuit = np.vstack([np.cos(theta), np.sin(theta)])
        
        # Joint Predictive Covariance Matrix for our two specific chosen slices
        Sigma = np.array([[var_x1, cov_x1_x2], [cov_x1_x2, var_x2]])
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
        
        # Rotate and scale the unit circle circuit into our confidence boundary ellipse (1.96 standard deviations)
        scaling_matrix = eigenvectors @ np.diag(1.96 * np.sqrt(np.maximum(eigenvalues, 1e-8)))
        ellipse_transformed = scaling_matrix @ ellipse_circuit
        
        # Shift the ellipse center from (0,0) to the model's actual predicted joint means!
        ellipse_x = ellipse_transformed[0, :] + mu_x1
        ellipse_y = ellipse_transformed[1, :] + mu_x2
        
        # --- Math: Generate the Bivariate Gaussian Background Mesh Grid ---
        y_min, y_max = mu_all.min() - 5.0, mu_all.max() + 5.0
        grid_resolution = 150
        y1_range = np.linspace(y_min, y_max, grid_resolution)
        y2_range = np.linspace(y_min, y_max, grid_resolution)
        Y1, Y2 = np.meshgrid(y1_range, y2_range)
        
        # Vectorized implementation of the Bivariate Gaussian PDF formula
        det_Sigma = var_x1 * var_x2 - cov_x1_x2**2
        inv_Sigma = np.array([[var_x2, -cov_x1_x2], [-cov_x1_x2, var_x1]]) / det_Sigma
        
        diff_y1 = Y1 - mu_x1
        diff_y2 = Y2 - mu_x2
        
        # Exponent evaluation matrix term: (y - mu)^T * Sigma^-1 * (y - mu)
        exponent = (diff_y1 * inv_Sigma[0, 0] + diff_y2 * inv_Sigma[1, 0]) * diff_y1 + \
                   (diff_y1 * inv_Sigma[0, 1] + diff_y2 * inv_Sigma[1, 1]) * diff_y2
                   
        pdf_z = (1.0 / (2 * np.pi * np.sqrt(np.maximum(det_Sigma, 1e-8)))) * np.exp(-0.5 * exponent)

        # --- Figure Assembly ---
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scatter3d"}, {"type": "scatter"}]],
            subplot_titles=(
                "1D Marginal Densities along Timeline (X-axis)",
                f"2D Marginal Density at (X1,X2)"
            )
        )
        
        # --- 1. Explicit Arrowed Axis for Time (X) ---
        time_min, time_max = X_min-1.0, X_max+1.0
        fig.add_trace(go.Scatter3d(
            x=[time_min, time_max], y=[0, 0], z=[0, 0],
            mode='lines', line=dict(color='black', width=4),
            name='Time Axis', showlegend=False
        ), row=1, col=1)
        
        fig.add_trace(go.Cone(
            x=[time_max], y=[0], z=[0],
            u=[1], v=[0], w=[0],
            sizemode="absolute", sizeref=(X_max - X_min) * 0.05,
            colorscale=[[0, 'black'], [1, 'black']], showscale=False,
            name='Time Arrow', showlegend=False
        ), row=1, col=1)

        # Plot the posterior on the x-y plane
        fig.add_trace(go.Scatter3d(
            x=X_line, y=mu_all, z=np.zeros_like(X_line),
            mode='lines', line=dict(color='green', width=5),
            name='GP Mean'
        ), row=1, col=1)

        # --- 2. Helper function to create a shifted, filled 3D Bell Curve slice ---
        def add_filled_bell_slice(fig, x_val, mu, variance, color, name):
            sigma = np.sqrt(variance)
            
            # Formulate dynamically centered Y bounds relative to this slice's unique mean
            y_pts = np.linspace(mu - 3.5 * sigma - 0.5, mu + 3.5 * sigma + 0.5, 100)
            z_density = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((y_pts - mu) / sigma)**2)
            N = len(y_pts)
            
            # Draw the solid contour wire across the top
            fig.add_trace(go.Scatter3d(
                x=np.full(N, x_val), y=y_pts, z=z_density,
                mode='lines', line=dict(color=color, width=4),
                name=name
            ), row=1, col=1)
            
            # Construct a triangular quad-mesh strip down to the floor (Z=0 plane)
            v_x = np.concatenate([np.full(N, x_val), np.full(N, x_val)])
            v_y = np.concatenate([y_pts, y_pts])
            v_z = np.concatenate([z_density, np.zeros(N)])
            
            i, j, k = [], [], []
            for idx in range(N - 1):
                i.append(idx)
                j.append(idx + 1)
                k.append(idx + N)
                i.append(idx + 1)
                j.append(idx + 1 + N)
                k.append(idx + N)
                
            fig.add_trace(go.Mesh3d(
                x=v_x, y=v_y, z=v_z,
                i=i, j=j, k=k,
                color=color, opacity=0.25,
                showlegend=False, hoverinfo='skip'
            ), row=1, col=1)

        # Draw the model's actual sliced densities
        add_filled_bell_slice(fig, x1, mu_x1, var_x1, 'rgb(239, 68, 68)', r'Slice X1 Posterior')
        add_filled_bell_slice(fig, x2, mu_x2, var_x2, 'rgb(68, 134, 239)', r'Slice X2 Posterior')
        
        # --- 3. Plot 2: Background Heatmap & Joint Ellipse view (Right Panel) ---
        # Add the bivariate density PDF value heatmap canvas underneath
        fig.add_trace(go.Heatmap(
            x=y1_range, y=y2_range, z=pdf_z,
            colorscale='Greens', showscale=True,
            colorbar=dict(
                title="Joint Density",
                x=1.02, y=0.5, len=0.8, thickness=15
            ),
            hoverinfo='z', name='PDF'
        ), row=1, col=2)

        # Draw the 95% Joint Confidence Ellipse contour lines over the density mapping
        fig.add_trace(go.Scatter(
            x=ellipse_x, y=ellipse_y, mode='lines',
            line=dict(color='red', width=2.5, dash='dash'),
            name='95% Joint Confidence'
        ), row=1, col=2)
        
        # Mark the moving joint mean coordinate position
        fig.add_trace(go.Scatter(
            x=[mu_x1], y=[mu_x2], mode='markers', marker=dict(color='black', size=12, symbol='x', line=dict(width=1)),
            name='Joint Mean Vector', showlegend=True
        ), row=1, col=2)
        
        # --- 4. Layout Configurations & Range Boundaries ---
        fig.update_layout(
            width=1200, height=550, template='plotly_white',
            title=f"GP Marginals at (X1,X2)=({x1:.1f}, {x2:.1f}): Mean (Y1,Y2)=({mu_x1:.2f}, {mu_x2:.2f}), Variances (σ₁², σ₂²)=({var_x1:.4f}, {var_x2:.4f}), Covariance={cov_x1_x2:.4f}",
            scene=dict(
                xaxis=dict(title="Time (X)", range=[time_min, time_max]),
                yaxis=dict(title="CO2 Value (Y)", range=[y_min, y_max]),
                zaxis=dict(title="Probability Density p(Y)", range=[-0.05, 1.2 / (np.sqrt(min(var_x1, var_x2)) * np.sqrt(2 * np.pi))]),
                camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)),
                aspectmode='manual',
                aspectratio=dict(x=1.2, y=1, z=0.6)
            ),
            xaxis_title=r"CO2 Value at Time X1 (Red)",
            yaxis_title=r"CO2 Value at Time X2 (Blue)",
            legend=dict(
                orientation="h",       # Horizontal layout
                yanchor="bottom",
                y=-0.3,                # Places the legend row perfectly above the subplots, below the main title
                xanchor="left",
                x=0.0
            )
        )
        
        fig.update_xaxes(range=[y_min, y_max], row=1, col=2)
        fig.update_yaxes(range=[y_min, y_max], row=1, col=2)

        with colab_output:
            clear_output(wait=True)
            fig.show()

    # Link callbacks to standard value properties
    slider_x1.observe(generate_gp_anatomy_plot, names='value')
    slider_x2.observe(generate_gp_anatomy_plot, names='value')

    dashboard_layout = widgets.VBox([
        widgets.HBox([slider_x1, slider_x2]),
        colab_output
    ])

    display(dashboard_layout)
    generate_gp_anatomy_plot()


# =================== DATA PREPARATION ========================
# Filter and downsample for interactive speed (Quarterly data)
TRAIN_START_YEAR, TRAIN_END_YEAR = 2000, 2020
NOW = datetime.now()
TEST_END_YEAR = NOW.year + 5

def datetime_to_decimal_year(dt: datetime) -> float:
    """Converts a datetime object to a continuous decimal year with minimal custom logic."""
    time_tuple = dt.timetuple()
    year = time_tuple.tm_year
    
    # Check if the specific year is a leap year (365 or 366 days)
    days_in_year = 366 if calendar.isleap(year) else 365
    
    # Calculate fractional day progress (including hours, minutes, seconds)
    # tm_yday is 1-indexed, so we subtract 1 for the start of the year fraction
    day_fraction = (time_tuple.tm_yday - 1) + (dt.hour / 24.0) + (dt.minute / 1440.0) + (dt.second / 86400.0)
    
    return year + (day_fraction / days_in_year)

def decimal_year_to_datetime(decimal_year: float) -> datetime:
    """Converts a continuous decimal year float back into a datetime object with minimal custom logic."""
    year = int(decimal_year)
    remainder = decimal_year - year
    
    # Dynamically resolve total days in the target year
    days_in_year = 366 if calendar.isleap(year) else 365
    
    # Map the fraction directly to total elapsed days
    elapsed_days = remainder * days_in_year
    
    # Offset from the exact beginning of the calendar year
    start_of_year = datetime(year, 1, 1)
    return start_of_year + timedelta(days=elapsed_days)


def prepare_CO2_dataset():
    print("Fetching NOAA Mauna Loa CO2 dataset...")
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"
    data = pd.read_csv(url, sep=r'\s+', comment='#', header=None,
                    names=['year', 'month', 'decimal_date', 'average', 'deseasonalized', 'ndays', 'sdev', 'unc'])
    data = data[['year', 'month', 'decimal_date', 'average']] # Keep only relevant columns

    df = data[(data['decimal_date'] >= TRAIN_START_YEAR) & (data['decimal_date'] <= datetime_to_decimal_year(NOW))]

    # Training data
    X_train = df['decimal_date'][df['decimal_date'] <= TRAIN_END_YEAR].values.reshape(-1, 1)
    y_train = df['average'][df['decimal_date'] <= TRAIN_END_YEAR].values
    # Testing data
    X_test = df['decimal_date'][df['decimal_date'] > TRAIN_END_YEAR].values.reshape(-1, 1)
    y_test = df['average'][df['decimal_date'] > TRAIN_END_YEAR].values
    # Future inference data (up to TEST_END_YEAR)
    X_future = np.linspace(datetime_to_decimal_year(NOW), TEST_END_YEAR, 300).reshape(-1, 1)
    # Datetime conversion for plotting and visualization
    X_train_dt = np.vectorize(decimal_year_to_datetime)(X_train)
    X_test_dt = np.vectorize(decimal_year_to_datetime)(X_test)
    X_future_dt = np.vectorize(decimal_year_to_datetime)(X_future)
    X_all = np.concatenate([X_train, X_test, X_future])
    X_all_dt = np.concatenate([X_train_dt, X_test_dt, X_future_dt])

    print(f"Data successfully loaded. Using {len(df)} data points out of {len(data)} total points.")
    print(f"Training data period: {X_train_dt[0][0].strftime('%Y/%m/%d')}-{X_train_dt[-1][0].strftime('%Y/%m/%d')}")
    print(f"Testing data period: {X_test_dt[0][0].strftime('%Y/%m/%d')}-{X_test_dt[-1][0].strftime('%Y/%m/%d')}")
    print(f"Future data period: {X_future_dt[0][0].strftime('%Y/%m/%d')}-{X_future_dt[-1][0].strftime('%Y/%m/%d')}")

    return data, X_train, X_train_dt, y_train, X_test_dt, y_test, X_all, X_all_dt

co2_data, X_train, X_train_dt, y_train, X_test_dt, y_test, X_all, X_all_dt = prepare_CO2_dataset()

# Partial update version is better but fails in Colab
# def show_rbf_example():
#     fig_naive = go.FigureWidget()
#     fig_naive.add_trace(go.Scatter(x=X_train_dt.flatten(), y=y_train, mode='markers', name='Training Data', marker=dict(color='black', size=4)))
#     fig_naive.add_trace(go.Scatter(x=X_test_dt.flatten(), y=y_test, mode='markers', name='Testing Data', marker=dict(color='blue', size=4)))
#     fig_naive.add_trace(go.Scatter(x=[], y=[], mode='lines', name='GP Mean', line=dict(color='red', width=2)))
#     fig_naive.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
#     fig_naive.add_trace(go.Scatter(x=[], y=[], mode='lines', fill='tonexty', fillcolor='rgba(255, 0, 0, 0.15)', line=dict(color='rgba(255,255,255,0)'), name='95% CI'))


#     fig_naive.update_layout(
#         title="Standalone RBF Kernel (Click on the legend to turn on/off each layer)",
#         xaxis_title="Year",
#         yaxis_title="CO2 (ppm)",
#         height=450,
#         width=1200,
#         template="plotly_white"
#     )

#     def update_naive(length_scale, variance):
#         # Construct a Scaled RBF kernel using a ConstantKernel multiplier
#         kernel = ConstantKernel(variance) * RBF(length_scale=length_scale)
        
#         # Pack into sklearn; optimizer=None locks hyperparameters to slider selections exactly
#         gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True, alpha=0.1)
#         gp.fit(X_train, y_train)
        
#         # Query prediction interval
#         mu, std = gp.predict(X_all, return_std=True)
        
#         # Synchronize Plotly graphics arrays
#         fig_naive.data[2].x = fig_naive.data[3].x = fig_naive.data[4].x = X_all_dt.flatten()
#         fig_naive.data[2].y = mu
#         fig_naive.data[3].y = mu - 1.96 * std # Lower Bound of 95% Confidence Interval
#         fig_naive.data[4].y = mu + 1.96 * std # Upper Bound of 95% Confidence Interval

#     # Create interactive sliders
#     slider_rbf_len = widgets.FloatSlider(value=10.0, min=1, max=50.0, step=0.1, description='Length Scale:', layout=widgets.Layout(width='500px'), style={'description_width': '100px'})
#     slider_rbf_var = widgets.FloatSlider(value=50.0, min=5.0, max=200.0, step=5.0, description='Constant Var:', layout=widgets.Layout(width='500px'), style={'description_width': '100px'})

#     # Synchronize layouts
#     ui_naive = widgets.HBox([slider_rbf_len, slider_rbf_var])
#     out_naive = widgets.interactive_output(update_naive, {'length_scale': slider_rbf_len, 'variance': slider_rbf_var})

#     display(ui_naive, fig_naive, out_naive)


def show_rbf_example():
    # Create interactive sliders
    slider_rbf_len = widgets.FloatSlider(value=10.0, min=1, max=50.0, step=0.1, description='Length Scale:', layout=widgets.Layout(width='500px'), style={'description_width': '100px'})
    slider_rbf_var = widgets.FloatSlider(value=50.0, min=5.0, max=200.0, step=5.0, description='Constant Var:', layout=widgets.Layout(width='500px'), style={'description_width': '100px'})
    ui_naive = widgets.HBox([slider_rbf_len, slider_rbf_var])
    out_plot = widgets.Output()

    def update_naive(length_scale, variance):
        with out_plot:
            clear_output(wait=True) # Clears the old plot before drawing the new one
            
            kernel = ConstantKernel(variance) * RBF(length_scale=length_scale)
            gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True, alpha=0.1)
            gp.fit(X_train, y_train)
            mu, std = gp.predict(X_all, return_std=True)
            
            # Rebuild standard figure (NOT FigureWidget)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X_train_dt.flatten(), y=y_train, mode='markers', name='Training Data', marker=dict(color='black', size=4)))
            fig.add_trace(go.Scatter(x=X_test_dt.flatten(), y=y_test, mode='markers', name='Testing Data', marker=dict(color='blue', size=4)))
            fig.add_trace(go.Scatter(x=X_all_dt.flatten(), y=mu, mode='lines', name='GP Mean', line=dict(color='red', width=2)))
            
            # Confidence intervals
            fig.add_trace(go.Scatter(x=X_all_dt.flatten(), y=mu - 1.96 * std, mode='lines', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
            fig.add_trace(go.Scatter(x=X_all_dt.flatten(), y=mu + 1.96 * std, mode='lines', fill='tonexty', fillcolor='rgba(255, 0, 0, 0.15)', line=dict(color='rgba(255,255,255,0)'), name='95% CI'))
            
            fig.update_layout(title="Standalone RBF Kernel (Click on the legend to turn on/off each layer)",
                              xaxis_title="Year",
                              yaxis_title="CO2 (ppm)",height=450, width=1200, template="plotly_white")
            fig.show()

    widgets.interactive_output(update_naive, {'length_scale': slider_rbf_len, 'variance': slider_rbf_var})
    
    display(ui_naive, out_plot)

# Same for this one
# def show_composite_example():

#     # Generate dummy data for pure sine wave and pure trend
#     X_dummy = np.linspace(-10, 10, 80).reshape(-1, 1)
#     y_sine = np.sin(X_dummy.flatten())/2 + np.random.normal(0, 0.05, 80)
#     y_trend = 1.05**(X_dummy.flatten())  + np.random.normal(0, 0.05, 80)
#     # COMBINED DATA: Sum of the periodic signal and the trend signal
#     y_combined = y_sine + y_trend

#     # Extended extrapolation test grid
#     X_test_dummy = np.linspace(-20, 20, 200).reshape(-1, 1)

#     # 2. SUBPLOTS CONFIGURATION: Change grid to 1 row by 3 columns
#     fig_dummy = go.FigureWidget(make_subplots(
#         rows=1, cols=3, 
#         subplot_titles=(
#             r"Periodic Kernel for y=sin(x)/2", 
#             r"RBF Kernel for y=1.05^x",
#             r"Composite Kernel (Sum) for y=sin(x)/2+1.05^x"
#         )
#     ))

#     # Subplot 1: Periodic
#     fig_dummy.add_trace(go.Scatter(x=X_dummy.flatten(), y=y_sine, mode='markers', marker=dict(color='black', size=4), name='Sine Data'), row=1, col=1)
#     fig_dummy.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color='blue'), name='Periodic Mean'), row=1, col=1)

#     # Subplot 2: RBF Trend
#     fig_dummy.add_trace(go.Scatter(x=X_dummy.flatten(), y=y_trend, mode='markers', marker=dict(color='black', size=4), name='Trend Data'), row=1, col=2)
#     fig_dummy.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color='green'), name='RBF Mean'), row=1, col=2)

#     # Subplot 3: Combined Data & Composite Fit (Indices 4 and 5 in fig_dummy.data)
#     fig_dummy.add_trace(go.Scatter(x=X_dummy.flatten(), y=y_combined, mode='markers', marker=dict(color='black', size=4), name='Combined Data'), row=1, col=3)
#     fig_dummy.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color='purple'), name='Composite Mean'), row=1, col=3)

#     fig_dummy.update_layout(
#         width=1200,
#         height=450, 
#         template="plotly_white", 
#         showlegend=True,
#         title="Visualization of ExpSineSquared, RBF, and Composite Additive Kernels on Dummy Data"
#     )

#     # Added per_len as a dynamic target variable mapped to the kernel parameters
#     def update_dummy(period, per_len, rbf_len):
#         # Fit Periodic
#         k_per = ExpSineSquared(length_scale=per_len, periodicity=period)
#         gp_per = GaussianProcessRegressor(kernel=k_per, optimizer=None, normalize_y=True, alpha=0.1)
#         gp_per.fit(X_dummy, y_sine)
#         fig_dummy.data[1].x = X_test_dummy.flatten()
#         fig_dummy.data[1].y = gp_per.predict(X_test_dummy)
        
#         # Fit RBF
#         k_rbf = RBF(length_scale=rbf_len)
#         gp_rbf = GaussianProcessRegressor(kernel=k_rbf, optimizer=None, normalize_y=True, alpha=0.5)
#         gp_rbf.fit(X_dummy, y_trend)
#         fig_dummy.data[3].x = X_test_dummy.flatten()
#         fig_dummy.data[3].y = gp_rbf.predict(X_test_dummy)
        
#         # Fit Composite Kernel via addition: (ExpSineSquared + RBF)
#         k_composite = ExpSineSquared(length_scale=per_len, periodicity=period) + RBF(length_scale=rbf_len)
#         gp_composite = GaussianProcessRegressor(kernel=k_composite, optimizer=None, normalize_y=True, alpha=0.2)
#         gp_composite.fit(X_dummy, y_combined)
#         fig_dummy.data[5].x = X_test_dummy.flatten()
#         fig_dummy.data[5].y = gp_composite.predict(X_test_dummy)

#     # Setup widgets container elements
#     w_period = widgets.FloatSlider(value=6.0, min=0.5, max=15.0, step=0.1, description='Periodic, Periodicity:', layout=widgets.Layout(width='400px'), style={'description_width': '150px'})
#     w_per_len = widgets.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Periodic, Length Scale:', layout=widgets.Layout(width='400px'), style={'description_width': '150px'})
#     w_rbf = widgets.FloatSlider(value=5, min=0.5, max=50.0, step=0.1, description='RBF Length Scale:', layout=widgets.Layout(width='400px'), style={'description_width': '150px'})

#     # Layout setup wrapping the instantiated configuration elements
#     ui_dummy = widgets.HBox([w_period, w_per_len, w_rbf])
#     out_dummy = widgets.interactive_output(update_dummy, {'period': w_period, 'per_len': w_per_len, 'rbf_len': w_rbf})

#     display(ui_dummy, fig_dummy, out_dummy)

def show_composite_example():

    # Generate dummy data for pure sine wave and pure trend
    X_dummy = np.linspace(-10, 10, 80).reshape(-1, 1)
    y_sine = np.sin(X_dummy.flatten())/2 + np.random.normal(0, 0.05, 80)
    y_trend = 1.05**(X_dummy.flatten())  + np.random.normal(0, 0.05, 80)
    # COMBINED DATA: Sum of the periodic signal and the trend signal
    y_combined = y_sine + y_trend

    # Extended extrapolation test grid
    X_test_dummy = np.linspace(-20, 20, 200).reshape(-1, 1)

    # Setup an explicit output widget to render the fresh figures safely
    out_plot = widgets.Output()

    # Added per_len as a dynamic target variable mapped to the kernel parameters
    def update_dummy(period, per_len, rbf_len):
        
        # Fit Periodic
        k_per = ExpSineSquared(length_scale=per_len, periodicity=period)
        gp_per = GaussianProcessRegressor(kernel=k_per, optimizer=None, normalize_y=True, alpha=0.1)
        gp_per.fit(X_dummy, y_sine)
        y_pred_per = gp_per.predict(X_test_dummy)
        
        # Fit RBF
        k_rbf = RBF(length_scale=rbf_len)
        gp_rbf = GaussianProcessRegressor(kernel=k_rbf, optimizer=None, normalize_y=True, alpha=0.5)
        gp_rbf.fit(X_dummy, y_trend)
        y_pred_rbf = gp_rbf.predict(X_test_dummy)
        
        # Fit Composite Kernel via addition: (ExpSineSquared + RBF)
        k_composite = ExpSineSquared(length_scale=per_len, periodicity=period) + RBF(length_scale=rbf_len)
        gp_composite = GaussianProcessRegressor(kernel=k_composite, optimizer=None, normalize_y=True, alpha=0.2)
        gp_composite.fit(X_dummy, y_combined)
        y_pred_composite = gp_composite.predict(X_test_dummy)

        # Clear previous plot and draw the new one cleanly inside the output widget context
        with out_plot:
            clear_output(wait=True)
            
            # Create a standard Figure (NOT a FigureWidget)
            fig_dummy = go.Figure(make_subplots(
                rows=1, cols=3, 
                subplot_titles=(
                    r"Periodic Kernel for y=sin(x)/2", 
                    r"RBF Kernel for y=1.05^x",
                    r"Composite Kernel (Sum) for y=sin(x)/2+1.05^x"
                )
            ))

            # Subplot 1: Periodic
            fig_dummy.add_trace(go.Scatter(x=X_dummy.flatten(), y=y_sine, mode='markers', marker=dict(color='black', size=4), name='Sine Data'), row=1, col=1)
            fig_dummy.add_trace(go.Scatter(x=X_test_dummy.flatten(), y=y_pred_per, mode='lines', line=dict(color='blue'), name='Periodic Mean'), row=1, col=1)

            # Subplot 2: RBF Trend
            fig_dummy.add_trace(go.Scatter(x=X_dummy.flatten(), y=y_trend, mode='markers', marker=dict(color='black', size=4), name='Trend Data'), row=1, col=2)
            fig_dummy.add_trace(go.Scatter(x=X_test_dummy.flatten(), y=y_pred_rbf, mode='lines', line=dict(color='green'), name='RBF Mean'), row=1, col=2)

            # Subplot 3: Combined Data & Composite Fit
            fig_dummy.add_trace(go.Scatter(x=X_dummy.flatten(), y=y_combined, mode='markers', marker=dict(color='black', size=4), name='Combined Data'), row=1, col=3)
            fig_dummy.add_trace(go.Scatter(x=X_test_dummy.flatten(), y=y_pred_composite, mode='lines', line=dict(color='purple'), name='Composite Mean'), row=1, col=3)

            fig_dummy.update_layout(
                width=1200,
                height=450, 
                template="plotly_white", 
                showlegend=True,
                title="Visualization of ExpSineSquared, RBF, and Composite Additive Kernels on Dummy Data"
            )

            # Use fig.show() instead of relying on display(fig_dummy)
            fig_dummy.show()

    # Setup widgets container elements
    w_period = widgets.FloatSlider(value=6.0, min=0.5, max=15.0, step=0.1, description='Periodic, Periodicity:', layout=widgets.Layout(width='400px'), style={'description_width': '150px'})
    w_per_len = widgets.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Periodic, Length Scale:', layout=widgets.Layout(width='400px'), style={'description_width': '150px'})
    w_rbf = widgets.FloatSlider(value=5, min=0.5, max=50.0, step=0.1, description='RBF Length Scale:', layout=widgets.Layout(width='400px'), style={'description_width': '150px'})

    # Connect the sliders to the update loop
    widgets.interactive_output(update_dummy, {'period': w_period, 'per_len': w_per_len, 'rbf_len': w_rbf})

    # Layout setup wrapping the instantiated configuration elements
    ui_dummy = widgets.HBox([w_period, w_per_len, w_rbf])
    
    # Display the sliders row and the output container housing the clean plot
    display(ui_dummy, out_plot)

def show_dashboard(build_kernel_func):


    # Set up the Main Dashboard Plot
    fig_main = go.FigureWidget()
    fig_main.add_trace(go.Scatter(x=X_train_dt.flatten(), y=y_train, mode='markers', name='Training data', marker=dict(color='black', size=4)))
    fig_main.add_trace(go.Scatter(x=X_test_dt.flatten(), y=y_test, mode='markers', name='Testing data', marker=dict(color='blue', size=4)))

    # Traces for Manual Tuning (Blue)
    fig_main.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Manual GP Mean', line=dict(color="#f63bbe", width=2, dash='dot')))
    fig_main.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
    fig_main.add_trace(go.Scatter(x=[], y=[], mode='lines', fill='tonexty', fillcolor='rgba(246, 59, 190, 0.15)', line=dict(color='rgba(246, 59, 190, 0.15)'), name='Manual 95% CI'))


    # Traces for LML Optimized Fit (Green)
    fig_main.add_trace(go.Scatter(x=[], y=[], mode='lines', name='LML Optimized Mean', line=dict(color='#10b981', width=2, dash='dash')))
    fig_main.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
    fig_main.add_trace(go.Scatter(x=[], y=[], mode='lines', fill='tonexty', fillcolor='rgba(16, 185, 129, 0.15)', line=dict(color='rgba(255,255,255,0)'), name='LML 95% CI'))


    fig_main.update_layout(title="Composite GP: Manual vs LML Optimization", xaxis_title="Year", yaxis_title="CO2 (ppm)",
                        height=450, width=1200, template="plotly_white")

    # Dashboard UI Elements
    sl_trend_v = widgets.FloatSlider(value=10.0, min=1.0, max=20.0, step=0.1, description='Trend: Variance', layout=widgets.Layout(width='400px'), style={'description_width': '150px'})
    sl_trend_l = widgets.FloatSlider(value=50.0, min=1.0, max=100.0, step=0.1, description='Trend: Length Scale', layout=widgets.Layout(width='400px'), style={'description_width': '150px'})
    sl_seasonal_v = widgets.FloatSlider(value=2.0, min=0.05, max=5.0, step=0.05, description='Seasonal: Variance', layout=widgets.Layout(width='400px'), style={'description_width': '150px'})
    sl_seasonal_l = widgets.FloatSlider(value=5.0, min=0.1, max=10.0, step=0.1, description='Seasonal: Length Scale', layout=widgets.Layout(width='400px'), style={'description_width': '150px'})
    sl_seasonal_p = widgets.FloatSlider(value=1.0, min=0.1, max=10.0, step=0.02, description='Seasonal: Periodicity', layout=widgets.Layout(width='400px'), style={'description_width': '150px'})
    sl_noise = widgets.FloatSlider(value=0.001, min=0.0001, max=0.1, step=0.0001, description='Noise: Variance', layout=widgets.Layout(width='400px'), style={'description_width': '150px'}, readout_format='.4f')

    btn_lml = widgets.Button(description="Optimize LML 🚀", button_style='success', icon='cogs', layout=widgets.Layout(width='200px'))
    lbl_lml_status = widgets.Label(value="")

    # HTML container to display optimized parameter output nicely
    html_lml_results = widgets.HTML(value="<div style='color: #64748b; font-style: italic; padding: 10px;'>Awaiting LML Optimization sequence...</div>")



    # Callback for Manual Sliders
    def update_manual_plot(change=None):
        kernel = build_kernel_func(sl_trend_v.value, sl_trend_l.value, sl_seasonal_v.value, sl_seasonal_l.value, sl_seasonal_p.value, sl_noise.value)
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True, alpha=0.0)
        gp.fit(X_train, y_train)
        
        mu, std = gp.predict(X_all, return_std=True)
        fig_main.data[2].x = fig_main.data[3].x = fig_main.data[4].x = X_all_dt.flatten()
        fig_main.data[2].y = mu
        fig_main.data[4].y = mu - 1.96 * std
        fig_main.data[3].y = mu + 1.96 * std

    # Link sliders to update function
    for sl in [sl_trend_v, sl_trend_l, sl_seasonal_v, sl_seasonal_l, sl_seasonal_p, sl_noise]:
        sl.observe(update_manual_plot, names='value')
    # Callback for LML Button (Robust Extraction via get_params())
    def run_lml_optimization(b):
        btn_lml.disabled = True
        lbl_lml_status.value = "Running L-BFGS-B Optimizer... please wait."
        
        # Use current slider positions as initial guesses
        kernel = build_kernel_func(sl_trend_v.value, sl_trend_l.value, sl_seasonal_v.value, sl_seasonal_l.value, sl_seasonal_p.value, sl_noise.value)
        
        # Let sklearn optimize the hyperparameters
        gp_opt = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, normalize_y=True, alpha=0.0)
        gp_opt.fit(X_train, y_train)
        
        mu_opt, std_opt = gp_opt.predict(X_all, return_std=True)
        
        # Render optimized lines
        fig_main.data[5].x = fig_main.data[6].x = fig_main.data[7].x = X_all_dt.flatten()
        fig_main.data[5].y = mu_opt
        fig_main.data[7].y = mu_opt - 1.96 * std_opt
        fig_main.data[6].y = mu_opt + 1.96 * std_opt
        
        # --- ROBUST EXTRACTION USING GET_PARAMS() ---
        params = gp_opt.kernel_.get_params()
        opt_trend_v = params['k1__k1__k1__constant_value']
        opt_trend_l = params['k1__k1__k2__length_scale']
        opt_seasonal_v = params['k1__k2__k1__constant_value']
        opt_seasonal_l = params['k1__k2__k2__length_scale']
        opt_seasonal_p = params['k1__k2__k2__periodicity']
        opt_noise = params['k2__noise_level']
        lml = gp_opt.log_marginal_likelihood()
        
        # Format a beautiful dashboard readout box
        html_lml_results.value = f"""
        <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 15px; font-family: sans-serif; max-width: 500px; box-shadow: 0 2px 8px rgba(0,0,0,0.02);">
            <h4 style="color: #166534; margin: 0 0 10px 0; font-weight: 700;">🎯 Mathematically Optimal Parameters (LML)</h4>
            <table style="width: 100%; font-size: 0.9rem; border-collapse: collapse; color: #1e293b;">
                <tr style="border-bottom: 1px solid #e2e8f0;"><td style="padding: 4px 0; font-weight:600;">Trend Variance:</td><td style="text-align: right; color:#059669; font-weight:700;">{opt_trend_v:.2f}</td></tr>
                <tr style="border-bottom: 1px solid #e2e8f0;"><td style="padding: 4px 0; font-weight:600;">Trend Lengthscale:</td><td style="text-align: right; color:#059669; font-weight:700;">{opt_trend_l:.2f}</td></tr>
                <tr style="border-bottom: 1px solid #e2e8f0;"><td style="padding: 4px 0; font-weight:600;">Seasonal Variance:</td><td style="text-align: right; color:#059669; font-weight:700;">{opt_seasonal_v:.2f}</td></tr>
                <tr style="border-bottom: 1px solid #e2e8f0;"><td style="padding: 4px 0; font-weight:600;">Seasonal Lengthscale:</td><td style="text-align: right; color:#059669; font-weight:700;">{opt_seasonal_l:.2f}</td></tr>
                <tr style="border-bottom: 1px solid #e2e8f0;"><td style="padding: 4px 0; font-weight:600;">Seasonal Periodicity:</td><td style="text-align: right; color:#059669; font-weight:700;">{opt_seasonal_p:.2f}</td></tr>
                <tr style="border-bottom: 1px solid #e2e8f0;"><td style="padding: 4px 0; font-weight:600;">Noise Level:</td><td style="text-align: right; color:#059669; font-weight:700;">{opt_noise:.4f}</td></tr>
                <tr style="margin-top: 5px;"><td style="padding: 6px 0 0 0; font-weight:700; color: #0f172a;">Max Log Likelihood:</td><td style="padding: 6px 0 0 0; text-align: right; color:#1e1b4b; font-weight:800;">{lml:.2f}</td></tr>
            </table>
        </div>
        """
        
        lbl_lml_status.value = ""
        btn_lml.disabled = False
    btn_lml.on_click(run_lml_optimization)

    # Initialize manual tracking lines
    update_manual_plot()

    # Layout
    ui_box_1 = widgets.VBox([
        widgets.HTML("<b style='font-size:1.1rem; color:#1e293b;'>1. Manual Hyperparameter Tuning:</b>"),
        widgets.HBox([sl_trend_v, sl_trend_l, sl_seasonal_v]),
        widgets.HBox([sl_seasonal_l, sl_seasonal_p, sl_noise])
    ])

    ui_box_2 = widgets.VBox([
        widgets.HTML("<b style='font-size:1.1rem; color:#1e293b;'>2. Automated Tuning (Log Marginal Likelihood Optimization):</b>"),
        widgets.HBox([
            widgets.VBox([btn_lml, lbl_lml_status], layout=widgets.Layout(margin='0 20px 0 0')), 
            html_lml_results
        ])
    ])

    display(ui_box_1, fig_main, ui_box_2)