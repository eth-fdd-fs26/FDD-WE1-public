import io
import base64
import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import HTML, display, clear_output

# True underlying functions for the Mars Rover mission
def f_star(X):
    x1, x2 = X[:, 0], X[:, 1]
    term1 = (x1**2 + x2 - 11.0) ** 2
    term2 = (x1 + x2**2 - 7.0) ** 2
    term3 = -5 * np.abs(x1 * x2)
    return 5*np.exp(-0.005 * (term1 + term2 + term3))
    
def g_star(X):
    x1, x2 = X[:, 0], X[:, 1]
    canyon = ((x1 + x2) ** 2) / 10.0
    boundary_decay = (x1**2 + x2**2) / 20.0
    return 1.5 - canyon - boundary_decay

def bump_function(center, radius, inf_val, max_val, X):
    """Vectorized bump function that takes `max_val` at `center` and decays to `inf_val` outside the `radius`.
    Note: Used to initialize the prior for the gp modelling the constraint g_star.
    """
    x1, x2 = center
    denom = (X[:, 0] - x1)**2 + (X[:, 1] - x2)**2 - radius**2
    mask = denom < -1e-10
    denom[~mask] = 1 # dummy nonzero value
    exponent = np.where(mask, 1/denom, -np.inf)
    return (max_val - inf_val) * np.exp( exponent + 1/radius**2) + inf_val

from __main__ import EXTENT, GRID_RES, GRID_RANGE, GRID_COORDS
def unflatten(arr):
    """Reshape a flattened array of shape (GRID_RES**2,) into a 2D array of shape (GRID_RES, GRID_RES)."""
    return arr.reshape(GRID_RES, GRID_RES)
def find_point_index(point):
    """Find the index of the selected point in the GRID_COORDS."""
    return np.where(np.isclose(GRID_COORDS, point, atol=1e-4).all(axis=-1))[0][0]
def observe(func, X, noise_var):
    """Observe the function at points X with Gaussian noise."""
    true_values = func(X)
    noise_values = true_values + np.random.normal(0, np.sqrt(noise_var), size=true_values.shape)
    return noise_values

TRUE_F_VALS = unflatten(f_star(GRID_COORDS)) # shape: (GRID_RES, GRID_RES) 
TRUE_G_VALS = unflatten(g_star(GRID_COORDS)) # shape: (GRID_RES, GRID_RES)



# Explanations & Quizzes
def explain_al_and_safebo():
    """Renders a beautifully formatted, intuitive markdown/HTML guide in the notebook 
    to clarify the distinction and synthesis between Active Learning and Safe BO.
    """
    html_content = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                max-width: 950px; margin: 25px auto; padding: 30px; background-color: #ffffff; 
                border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 10px 30px rgba(0,0,0,0.02); color: #334155;">
        
        <div style="border-bottom: 2px solid #f1f5f9; padding-bottom: 15px; margin-bottom: 25px;">
            <span style="background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">📖 Conceptual Briefing</span>
            <h2 style="margin: 8px 0 0 0; color: #0f172a; font-size: 1.6rem; font-weight: 800;">Demystifying Active Learning vs. Safe Bayesian Optimization</h2>
        </div>

        <div style="margin-bottom: 25px;">
            <h4 style="color: #2563eb; margin: 0 0 8px 0; font-size: 1.15rem; font-weight: 700;">1. Transductive Active Learning: "Mapping the Unknown Efficiently"</h4>
            <p style="font-size: 0.95rem; line-height: 1.6; margin: 0; color: #475569;">
                Imagine you are dropped onto an unfamiliar landscape on Mars. You have a finite amount of battery power, meaning you can only pick a small number of physical locations to drill and sample. 
                Instead of picking points randomly or executing a wasteful grid-search, <strong>Active Learning (AL)</strong> lets the machine choose its own training data. 
                The word <em>transductive</em> simply means we only care about doing an excellent job mapping a specific target region rather than trying to generalize to the entire planet. 
                <br><br>
                💡 <strong>The Goal:</strong> Maximize information gain. The rover actively seeks out areas where its model has the highest uncertainty (<math display="inline"><mi>&sigma;</mi></math>) to eliminate blind spots as quickly as possible.
            </p>
        </div>

        <div style="margin-bottom: 25px;">
            <h4 style="color: #dc2626; margin: 0 0 8px 0; font-size: 1.15rem; font-weight: 700;">2. Safe Bayesian Optimization: "Maximizing Rewards Without Dying"</h4>
            <p style="font-size: 0.95rem; line-height: 1.6; margin: 0; color: #475569;">
                Standard Bayesian Optimization (BO) is single-minded: it searches for the highest peak of a function (e.g., finding the maximum concentration of a rare mineral). 
                However, in the real world, exploring blindly can be fatal. If your rover drives over a cliff or sinks into a sand trap, the mission instantly fails. 
                <strong>Safe BO</strong> introduces a constraint <math display="inline"><msup><mi>g</mi><mo>*</mo></msup><mo>(</mo><mi>X</mi><mo>)</mo><mo>&ge;</mo><mn>0</mn></math> alongside our target function <math display="inline"><msup><mi>f</mi><mo>*</mo></msup><mo>(</mo><mi>X</mi><mo>)</mo></math>. 
                It uses the Gaussian Process safety confidence bounds to guarantee that the rover <em>only</em> samples coordinates it is highly confident are physically safe.
            </p>
        </div>

        <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 20px; border-radius: 12px;">
            <h4 style="color: #0d9488; margin: 0 0 10px 0; font-size: 1.15rem; font-weight: 700;">3. The Synthesis: Our Mission Pipeline</h4>
            <p style="font-size: 0.95rem; line-height: 1.6; margin: 0; color: #475569;">
                In this exercise, we weave these two frameworks together into a cohesive loop. The rover balances safety boundaries with ambitious resource discovery by partitioning the landscape into specialized spaces:
            </p>
            <ul style="font-size: 0.92rem; line-height: 1.6; color: #475569; padding-left: 20px; margin-top: 10px; margin-bottom: 0;">
                <li style="margin-bottom: 8px;"><strong>Safe BO draws the boundary:</strong> First, the rover uses the safety posterior to calculate a safe zone which it never steps outside.</li>
                <li style="margin-bottom: 8px;"><strong>Safe BO identifies the target region:</strong> Next, it flags points that have a realistic chance of being better than the current highest mineral peak found so far.</li>
                <li style="margin-bottom: 0;"><strong>Active Learning picks the drill site:</strong> Finally, Active Learning steps in and selects a safe point where the <em>uncertainty about the target region is highest</em>. This is a directed way to explore the most informative areas while maintaining safety.</li>
            </ul>
        </div>
        
    </div>
    """
    return HTML(html_content)

def show_introduction():
    """Generates a responsive HTML/CSS introductory panel for the Mars Rover 
    mining mission using native HTML5 MathML for formulas.
    """
    html_content = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
                max-width: 900px; margin: 20px auto; padding: 30px; background-color: #f8fafc; 
                border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 20px rgba(0,0,0,0.03);
                color: #334155;">
        
        <div style="text-align: center; margin-bottom: 25px;">
            <span style="background: #e0f2fe; color: #0369a1; padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
                🚀 Mission Briefing
            </span>
            <h2 style="color: #1e293b; margin-top: 10px; margin-bottom: 0; font-size: 1.6rem; font-weight: 800; letter-spacing: -0.025em;">
                Mars Rover Autonomous Exploration
            </h2>
        </div>
        
        <p style="font-size: 0.95rem; line-height: 1.6; margin-top: 0; margin-bottom: 20px; color: #475569;">
            You are the head of a space-mining company, and you are in charge of navigating a Mars Rover on a mining mission.
        </p>
        
        <p style="font-size: 0.95rem; line-height: 1.6; margin-bottom: 25px; color: #475569;">
            Based on observations made from Earth, your team has identified a massive region 
            <math display="inline"><mi>R</mi></math> on Mars that shows strong signs of a highly valuable, 
            deep-crust mineral vein. Because the region is far too large for a full excavation, your team has deployed a 
            rover to narrow down the single most promising extraction site. Your objective is to find the absolute peak 
            concentration of this mineral by commanding the rover to drill and sample specific coordinates.
        </p>
        
        <h4 style="color: #1e293b; font-size: 1.1rem; margin-top: 0; margin-bottom: 12px; font-weight: 700;">
            ⚠️ Critical Mission Constraints
        </h4>
        <div style="display: flex; flex-direction: column; gap: 15px; margin-bottom: 25px;">
            
            <div style="display: flex; gap: 15px; background: white; padding: 15px; border-radius: 12px; border-left: 5px solid #f59e0b; box-shadow: 0 2px 8px rgba(0,0,0,0.01);">
                <div style="background: #fef3c7; color: #b45309; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; flex-shrink: 0;">⏳</div>
                <div>
                    <h5 style="margin: 0 0 4px 0; color: #1e293b; font-size: 1rem; font-weight: 600;">Resource Limits</h5>
                    <p style="margin: 0; color: #64748b; font-size: 0.9rem; line-height: 1.4;">
                        Core drilling is incredibly time-consuming and expensive. You must find the maximum value using as few physical trials as possible.
                    </p>
                </div>
            </div>
            
            <div style="display: flex; gap: 15px; background: white; padding: 15px; border-radius: 12px; border-left: 5px solid #ef4444; box-shadow: 0 2px 8px rgba(0,0,0,0.01);">
                <div style="background: #fee2e2; color: #b91c1c; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; flex-shrink: 0;">💥</div>
                <div>
                    <h5 style="margin: 0 0 4px 0; color: #1e293b; font-size: 1rem; font-weight: 600;">Safety Issues</h5>
                    <p style="margin: 0; color: #64748b; font-size: 0.9rem; line-height: 1.4;">
                        The ground is highly volatile, and drilling into an unstable location will trigger a structural failure, destroying the rover and ending the mission immediately.
                    </p>
                </div>
            </div>
            
        </div>
        
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; padding: 15px 20px; border-radius: 12px; text-align: left;">
            <p style="margin: 0; color: #166534; font-size: 0.95rem; line-height: 1.5; font-weight: 500;">
                To succeed, you must master a delicate balance: aggressively hunting for the ultimate geological treasure 
                while respecting the absolute necessity of staying alive. This means that your Rover has to act in a more 
                <span style="font-weight: 700; color: #15803d; font-style: italic;">directed</span> manner and 
                <span style="font-weight: 700; color: #15803d; font-style: italic;">extrapolate</span> beyond the limited information.
            </p>
        </div>
        
    </div>
    """
    return HTML(html_content)


def show_terrain_info():
    """Generates a responsive HTML/CSS panel describing the exploration grid
    and dual-function mapping using native HTML5 MathML for formulas.
    """
    html_content = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
                max-width: 900px; margin: 20px auto; padding: 30px; background-color: #f8fafc; 
                border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 20px rgba(0,0,0,0.03);
                color: #334155;">
        
        <div style="text-align: center; margin-bottom: 25px;">
            <span style="background: #f3e8ff; color: #6b21a8; padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
                🗺️ Grid & Operational Domain
            </span>
            <h2 style="color: #1e293b; margin-top: 10px; margin-bottom: 0; font-size: 1.6rem; font-weight: 800; letter-spacing: -0.025em;">
                Terrain Mapping & Boundaries
            </h2>
        </div>
        
        <p style="font-size: 0.95rem; line-height: 1.6; margin-top: 0; margin-bottom: 25px; color: #475569; text-align: left;">
            The terrain of interest is mapped as a continuous coordinate square 
            <math display="inline">
                <mi>R</mi><mo>=</mo><mo>[</mo><mo>-</mo><mn>5</mn><mo>,</mo><mn>5</mn><mo>]</mo><mo>&times;</mo><mo>[</mo><mo>-</mo><mn>5</mn><mo>,</mo><mn>5</mn><mo>]</mo>
            </math>, 
            discretized evenly into grids of side length 
            <math display="inline"><mn>0.1</mn></math>. Your Rover will drill at the corners of these grid cells.
        </p>
        
        <h4 style="color: #1e293b; font-size: 1.1rem; margin-top: 0; margin-bottom: 15px; font-weight: 700;">
            📡 Two Defining Properties to Model
        </h4>
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            
            <div style="flex: 1; min-width: 280px; background: white; padding: 20px; border-radius: 12px; border-top: 5px solid #3b82f6; box-shadow: 0 2px 8px rgba(0,0,0,0.01); display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <h5 style="margin: 0 0 10px 0; color: #1e293b; font-size: 1.05rem; font-weight: 700; display: flex; align-items: center; justify-content: space-between;">
                        <span>Objective Function</span>
                        <math display="inline"><msup><mi>f</mi><mo>*</mo></msup></math>
                    </h5>
                    <p style="margin: 0; color: #64748b; font-size: 0.9rem; line-height: 1.5;">
                        Represents the underlying true mineral concentration across the map region. Your optimization target is to find coordinates that maximize this value.
                    </p>
                </div>
                <div style="margin-top: 15px; background: #eff6ff; padding: 6px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; color: #1d4ed8; text-align: center; text-transform: uppercase; letter-spacing: 0.025em;">
                    Target: Maximize Output
                </div>
            </div>
            
            <div style="flex: 1; min-width: 280px; background: white; padding: 20px; border-radius: 12px; border-top: 5px solid #10b981; box-shadow: 0 2px 8px rgba(0,0,0,0.01); display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <h5 style="margin: 0 0 10px 0; color: #1e293b; font-size: 1.05rem; font-weight: 700; display: flex; align-items: center; justify-content: space-between;">
                        <span>Safety Function</span>
                        <math display="inline"><msup><mi>g</mi><mo>*</mo></msup></math>
                    </h5>
                    <div style="margin: 0; color: #64748b; font-size: 0.9rem; line-height: 1.5;">
                        Models the regional crust structural stability. It is safe for the rover to navigate and perform sample drilling operations only within the valid safe set region:
                        
                        <div style="margin-top: 10px; background: #f0fdf4; padding: 8px; border-radius: 6px; text-align: center;">
                            <math display="block">
                                <msup><mi>S</mi><mo>*</mo></msup>
                                <mo>:=</mo>
                                <mrow><mo>{</mo>
                                    <mi>X</mi><mo>&in;</mo><mi>R</mi><mo>:</mo><msup><mi>g</mi><mo>*</mo></msup><mo>(</mo><mi>X</mi><mo>)</mo><mo>&ge;</mo><mn>0</mn>
                                <mo>}</mo></mrow>
                            </math>
                        </div>
                    </div>
                </div>
                <div style="margin-top: 15px; background: #ecfdf5; padding: 6px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; color: #047857; text-align: center; text-transform: uppercase; letter-spacing: 0.025em;">
                    Bound: Strict Constraint
                </div>
            </div>
            
        </div>
    </div>
    """
    return HTML(html_content)


def safe_bo_workflow():
    """Generates a responsive HTML/CSS workflow component for the Safe Bayesian
    Optimization and Transductive Active Learning loop, using native HTML5 MathML 
    for mathematical formulas (no MathJax required).
    """
    html_content = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
                max-width: 900px; margin: 20px auto; padding: 25px; background-color: #f8fafc; 
                border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 20px rgba(0,0,0,0.03);">
        
        <h3 style="text-align: center; color: #1e293b; margin-top: 0; margin-bottom: 25px; font-size: 1.4rem; font-weight: 700;">
            🏗️ Safe Bayesian Optimization Workflow
        </h3>
        
        <div style="display: flex; flex-direction: column; gap: 20px; position: relative;">
            
            <div style="display: flex; gap: 15px; background: white; padding: 15px; border-radius: 12px; border-left: 5px solid #22c55e; box-shadow: 0 2px 8px rgba(0,0,0,0.01);">
                <div style="background: #dcfce7; color: #15803d; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0;">1</div>
                <div>
                    <h4 style="margin: 0 0 5px 0; color: #1e293b; font-size: 1.05rem; display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
                        Initial Safe Point 
                    </h4>
                    <p style="margin: 0; color: #64748b; font-size: 0.9rem; line-height: 1.4; display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
                        The mission kicks off at a point <math display="inline"><msub><mi>X</mi><mn>0</mn></msub><mo>∈</mo><msup><mi>S</mi><mo>*</mo></msup></math> where the crust is guaranteed to be stable, i.e. 
                        <math display="inline"><msup><mi>g</mi><mo>*</mo></msup><mo>(</mo><msub><mi>X</mi><mn>0</mn></msub><mo>)</mo><mo>≥</mo><mn>0</mn></math>.
                    </p>
                </div>
            </div>
            
            <div style="display: flex; gap: 15px; background: white; padding: 15px; border-radius: 12px; border-left: 5px solid #f59e0b; box-shadow: 0 2px 8px rgba(0,0,0,0.01);">
                <div style="background: #fef3c7; color: #b45309; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0;">2</div>
                <div>
                    <h4 style="margin: 0 0 5px 0; color: #1e293b; font-size: 1.05rem; display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
                        Noisy Physical Telemetry Collection
                    </h4>
                    <div style="margin: 0; color: #64748b; font-size: 0.9rem; line-height: 1.4;">
                        At round <math display="inline"><mi>n</mi><mo>≥</mo><mn>0</mn></math>, drilling at site <math display="inline"><msub><mi>X</mi><mi>n</mi></msub></math> returns hardware-corrupted telemetry. Measurement errors are explicitly modeled as zero-mean Gaussian independent noise:

                        <div style="margin-top: 8px; background: #f1f5f9; padding: 12px; border-radius: 6px; color: #334155; text-align: center;">
                            <math display="block">
                                <msub><mi>y</mi><mi>n</mi></msub><mo>=</mo><msup><mi>f</mi><mo>*</mo></msup><mo>(</mo><msub><mi>X</mi><mi>n</mi></msub><mo>)</mo><mo>+</mo><msub><mi>ϵ</mi><mi>X</mi></msub>,
                                <mspace width="6em"></mspace>
                                <msub><mi>z</mi><mi>n</mi></msub><mo>=</mo><msup><mi>g</mi><mo>*</mo></msup><mo>(</mo><msub><mi>X</mi><mi>n</mi></msub><mo>)</mo><mo>+</mo><msub><mi>η</mi><mi>X</mi></msub>
                            </math>
                        </div>
                    </div>
                </div>
            </div>
            
            <div style="display: flex; gap: 15px; background: white; padding: 15px; border-radius: 12px; border-left: 5px solid #3b82f6; box-shadow: 0 2px 8px rgba(0,0,0,0.01);">
                <div style="background: #dbeafe; color: #1d4ed8; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0;">3</div>
                <div style="flex: 1; min-width: 0;"> <h4 style="margin: 0 0 5px 0; color: #1e293b; font-size: 1.05rem; display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
                        Dataset Accumulation 
                    </h4>
                    <div style="margin: 0; color: #64748b; font-size: 0.9rem; line-height: 1.4;">
                        Add new observations for concentration value <math display="inline"><msub><mi>y</mi><mi>n</mi></msub></math> and safety margin <math display="inline"><msub><mi>z</mi><mi>n</mi></msub></math> to existing datasets
                    
                        <div style="margin-top: 8px; background: #f1f5f9; padding: 12px; border-radius: 6px; color: #334155; text-align: center;">
                            <math display="block">
                                <msub><mi>D</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>:=</mo><msubsup><mrow><mo>{</mo><msub><mi>X</mi><mi>i</mi></msub><mo>}</mo></mrow><mrow><mi>i</mi><mo>=</mo><mn>0</mn></mrow><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msubsup>
                                
                                <mspace width="6em"></mspace>
                                <msubsup><mi>D</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow><mi>f</mi></msubsup><mo>:=</mo><msubsup><mrow><mo>{</mo><mo>(</mo><msub><mi>X</mi><mi>i</mi></msub><mo>,</mo><msub><mi>y</mi><mi>i</mi></msub><mo>)</mo><mo>}</mo></mrow><mrow><mi>i</mi><mo>=</mo><mn>0</mn></mrow><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msubsup>
                                
                                <mspace width="6em"></mspace>
                                <msubsup><mi>D</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow><mi>g</mi></msubsup><mo>:=</mo><msubsup><mrow><mo>{</mo><mo>(</mo><msub><mi>X</mi><mi>i</mi></msub><mo>,</mo><msub><mi>z</mi><mi>i</mi></msub><mo>)</mo><mo>}</mo></mrow><mrow><mi>i</mi><mo>=</mo><mn>0</mn></mrow><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msubsup>
                            </math>
                        </div>

                        <p style="margin: 8px 0; color: #64748b; font-size: 0.9rem; line-height: 1.4;">and obtain</p>
                        
                        <div style="margin-top: 8px; background: #f1f5f9; padding: 12px; border-radius: 6px; color: #334155; text-align: center;">
                            <math display="block">
                                <msub><mi>D</mi><mi>n</mi></msub><mo>&leftarrow;</mo><msub><mi>D</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>&cup;</mo><msubsup><mrow><mo>{</mo><msub><mi>X</mi><mi>n</mi></msub><mo>}</mo></mrow></msubsup>
                                
                                <mspace width="6em"></mspace>
                                <msubsup><mi>D</mi><mi>n</mi><mi>f</mi></msubsup><mo>&leftarrow;</mo><msubsup><mi>D</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow><mi>f</mi></msubsup><mo>&cup;</mo><mrow><mo>{</mo><mo>(</mo><msub><mi>X</mi><mi>n</mi></msub><mo>,</mo><msub><mi>y</mi><mi>n</mi></msub><mo>)</mo><mo>}</mo></mrow>
                                
                                <mspace width="6em"></mspace>
                                <msubsup><mi>D</mi><mi>n</mi><mi>g</mi></msubsup><mo>&leftarrow;</mo><msubsup><mi>D</mi><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow><mi>g</mi></msubsup><mo>&cup;</mo><mrow><mo>{</mo><mo>(</mo><msub><mi>X</mi><mi>n</mi></msub><mo>,</mo><msub><mi>z</mi><mi>n</mi></msub><mo>)</mo><mo>}</mo></mrow>
                            </math>
                        </div>                    
                    </div>
                </div>
            </div>

            <div style="display: flex; gap: 15px; background: white; padding: 15px; border-radius: 12px; border-left: 5px solid #8b5cf6; box-shadow: 0 2px 8px rgba(0,0,0,0.01);">
                <div style="background: #ede9fe; color: #6d28d9; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0;">4</div>
                <div>
                    <h4 style="margin: 0 0 5px 0; color: #1e293b; font-size: 1.05rem;">Gaussian Process Updates</h4>
                    
                    <p style="margin: 0; color: #64748b; font-size: 0.9rem; line-height: 1.4;">
                    Based on all the past observations, update your beliefs about the unknown functions <math display="inline"><msup><mi>f</mi><mo>*</mo></msup></math> and <math display="inline"><msup><mi>g</mi><mo>*</mo></msup></math> by taking the posterior distributions of the priors
                    <math display="inline">
                        <mi>f</mi><mspace width="0.5em"></mspace><mo>~</mo><mspace width="0.5em"></mspace><mtext>GP</mtext><mo>(</mo><mi>&mu;</mi><mo>,</mo><mi>k</mi><mo>)</mo>
                    </math> 
                    and 
                    <math display="inline">
                        <mi>g</mi><mspace width="0.5em"></mspace><mo>~</mo><mspace width="0.5em"></mspace><mtext>GP</mtext><mo>(</mo><msup><mi>&mu;</mi><mo>&prime;</mo></msup><mo>,</mo><msup><mi>k</mi><mo>&prime;</mo></msup><mo>)</mo>
                    </math>:
                    </p>
                    <div style="margin-top: 8px; background: #f1f5f9; padding: 12px; border-radius: 6px; color: #334155; text-align: center;">
                        <math display="block">
                            <mi>f</mi>
                            <mo>|</mo>
                            <msubsup><mi>D</mi><mi>n</mi><mi>f</mi></msubsup>
                            <mspace width="0.5em"></mspace>
                            <mo>~</mo>
                            <mspace width="0.5em"></mspace>
                            <mtext>GP</mtext>
                            <mo>(</mo>
                            <msub><mi>&mu;</mi><mi>n</mi></msub>
                            <mo>,</mo>
                            <msub><mi>k</mi><mi>n</mi></msub>
                            <mo>)</mo>
                            
                            
                            <mspace width="6em"></mspace>
                            
                            <mi>g</mi>
                            <mo>|</mo>
                            <msubsup><mi>D</mi><mi>n</mi><mi>g</mi></msubsup>
                            <mspace width="0.5em"></mspace>
                            <mo>~</mo>
                            <mspace width="0.5em"></mspace>
                            <mtext>GP</mtext>
                            <mo>(</mo>
                            <msubsup><mi>&mu;</mi><mi>n</mi><mo>&prime;</mo></msubsup>
                            <mo>,</mo>
                            <msubsup><mi>k</mi><mi>n</mi><mo>&prime;</mo></msubsup>
                            <mo>)</mo>
                        </math>
                    </div>
                </div>
            </div>
            
            <div style="display: flex; gap: 15px; background: white; padding: 15px; border-radius: 12px; border-left: 5px solid #ec4899; box-shadow: 0 2px 8px rgba(0,0,0,0.01);">
                <div style="background: #fce7f3; color: #be185d; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0;">5</div>
                <div>
                    <h4 style="margin: 0 0 5px 0; color: #1e293b; font-size: 1.05rem;">Information-Based Transductive Active Learning Decision Rule</h4>
                    <p style="margin: 0; color: #64748b; font-size: 0.9rem; line-height: 1.4; display: flex; flex-direction: column;">
                        <span>Based on your updated beliefs, select the next point <math display="inline"><msub><mi>X</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub></math> which you believe to lie inside
                        <math display="inline"><msup><mi>S</mi><mo>*</mo></msup></math> and yield maximal knowledge gain in a promising area.</span>
                    </p>
                </div>
            </div>
            
        </div>
        
        <div style="margin-top: 25px; text-align: center; border-top: 1px dashed #cbd5e1; padding-top: 15px;">
            <span style="background: #f1f5f9; color: #475569; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; display: inline-flex; align-items: center; gap: 4px;">
                🔄 Loop Back to Step 2 for Round <math display="inline"><mi>n</mi><mo>+</mo><mn>1</mn></math>
            </span>
        </div>
    </div>
    """
    return HTML(html_content)



def visualize_region_of_interest(figsize=(12,5)):
    """Generates the true underlying terrain maps inside a high-fidelity 
    mission telemetry HTML container card.
    """
    # 1. Create the Matplotlib figure using clean parameters
    fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=110)
    
    # Render maps with consistent color limits
    im1 = ax[0].imshow(TRUE_F_VALS, origin='lower', extent=EXTENT, cmap='coolwarm', 
                       vmin=0, vmax=TRUE_F_VALS.max() + 0.5) 
    im2 = ax[1].imshow(TRUE_G_VALS, origin='lower', extent=EXTENT, cmap='coolwarm', 
                       vmin=TRUE_G_VALS.min() - 0.5, vmax=TRUE_G_VALS.max() + 0.5) 
    
    # Safety boundary contour
    cntr = ax[1].contour(TRUE_G_VALS, levels=[0], colors='black', extent=EXTENT, linewidths=1.5)
    
    # Add cleanly labeled colorbars
    fig.colorbar(im1, ax=ax[0], orientation='vertical', label="Mineral Concentration")
    fig.colorbar(im2, ax=ax[1], orientation='vertical', label="Safety Level")
    
    # Legend formatting using native MathML alignment fallback text
    ax[1].legend([cntr.legend_elements()[0][0]], ["g* = 0"], loc='upper right', framealpha=0.9)
    
    # Subplot styling
    ax[0].set_title("True Mineral Concentration (f*)", fontsize=11, color='#475569', weight='600', pad=10)
    ax[1].set_title("True Safety Level (g*)", fontsize=11, color='#475569', weight='600', pad=10)
    
    for a in ax:
        a.set_aspect('equal')
        a.tick_params(colors='#64748b', labelsize=9)
        for spine in a.spines.values():
            spine.set_color('#cbd5e1')
            
    plt.tight_layout()
    
    # 2. Save the figure into an in-memory binary byte stream
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    
    # 3. Encode image bytes to base64 text
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)  # Prevents raw plot duplication leaking outside the HTML wrapper
    
    # 4. Embed the image inside our unified dashboard CSS container template
    html_wrapper = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
                max-width: 900px; margin: 20px auto; padding: 25px; background-color: #f8fafc; 
                border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 20px rgba(0,0,0,0.03);">
        
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="background: #f1f5f9; color: #475569; padding: 5px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
                🏗️ Terrain Ground Truth
            </span>
            <h3 style="color: #1e293b; margin-top: 6px; margin-bottom: 0; font-size: 1.3rem; font-weight: 700;">
                Properties of the Region of Interest <math display="inline"><mi>R</mi></math>
            </h3>
        </div>
        
        <div style="background: white; padding: 15px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.01); text-align: center;">
            <img src="data:image/png;base64,{img_base64}" style="max-width: 100%; height: auto; display: block; margin: 0 auto; border-radius: 6px;" alt="Mars Rover Ground Truth Matrix Maps"/>
        </div>
        
        <div style="margin-top: 15px; text-align: center;">
            <p style="margin: 0; font-size: 0.9rem; color: red; font-style: italic;">
                ⚠️ These underlying values are unknown to you and the Rover!
            </p>
        </div>
    </div>
    """
    return HTML(html_wrapper)



def gp_priors_quiz():
    """Generates an interactive, responsive multiple-choice quiz about 
    choosing Gaussian Process prior means for safe active learning.
    """
    # Unique ID for tracking choices inside the notebook cell
    uid = "quiz_gp_priors"
    
    html_content = f"""
    <div id="{uid}" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
                max-width: 900px; margin: 25px auto; padding: 25px; background-color: #f8fafc; 
                border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 20px rgba(0,0,0,0.03); color: #334155;">
        
        <div style="margin-bottom: 20px;">
            <span style="background: #e0f2fe; color: #0369a1; padding: 5px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
                🧠 Knowledge Check
            </span>
            <h4 style="color: #1e293b; margin-top: 10px; margin-bottom: 8px; font-size: 1.1rem; font-weight: 700; line-height: 1.4;">
                To conduct learning using Gaussian Processes, you need to specify the prior distributions. What values should we take as the prior means 
                <math display="inline"><mi>&mu;</mi></math> for 
                <math display="inline"><msup><mi>&mu;</mi><mo>&prime;</mo></msup></math> and 
                <math display="inline"><mi>f</mi></math> for 
                <math display="inline"><mi>g</mi></math>?
            </h4>
        </div>
        
        <div style="background: #fffbeb; border: 1px solid #fde68a; padding: 12px 15px; border-radius: 10px; margin-bottom: 20px; font-size: 0.88rem; line-height: 1.4; color: #92400e;">
            <strong>💡 Hint:</strong> The prior mean <math display="inline"><mi>&mu;</mi></math> tells you which area is promising, which you leverage to balance exploration (regions with high uncertainty) and exploitation (regions with low uncertainty but high known values). The prior mean <math display="inline"><msup><mi>&mu;</mi><mo>&prime;</mo></msup></math> tells you which area is dangerous, and you do not want to misjudge.
        </div>
        
        <div style="display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px;">
            <label style="display: flex; align-items: center; gap: 10px; background: white; padding: 12px 15px; border: 1px solid #e2e8f0; border-radius: 10px; cursor: pointer; transition: background 0.2s;">
                <input type="radio" name="gp_prior_opt" value="1" style="accent-color: #2563eb; transform: scale(1.1); cursor: pointer;">
                <span style="font-size: 0.92rem;">Set both <math display="inline"><mi>&mu;</mi></math> and <math display="inline"><msup><mi>&mu;</mi><mo>&prime;</mo></msup></math> to zero</span>
            </label>
            
            <label style="display: flex; align-items: center; gap: 10px; background: white; padding: 12px 15px; border: 1px solid #e2e8f0; border-radius: 10px; cursor: pointer; transition: background 0.2s;">
                <input type="radio" name="gp_prior_opt" value="2" style="accent-color: #2563eb; transform: scale(1.1); cursor: pointer;">
                <span style="font-size: 0.92rem;">Set both <math display="inline"><mi>&mu;</mi></math> and <math display="inline"><msup><mi>&mu;</mi><mo>&prime;</mo></msup></math> to a large positive value</span>
            </label>
            
            <label style="display: flex; align-items: center; gap: 10px; background: white; padding: 12px 15px; border: 1px solid #e2e8f0; border-radius: 10px; cursor: pointer; transition: background 0.2s;">
                <input type="radio" name="gp_prior_opt" value="3" style="accent-color: #2563eb; transform: scale(1.1); cursor: pointer;">
                <span style="font-size: 0.92rem;">Set both <math display="inline"><mi>&mu;</mi></math> and <math display="inline"><msup><mi>&mu;</mi><mo>&prime;</mo></msup></math> to a negative value</span>
            </label>
            
            <label style="display: flex; align-items: center; gap: 10px; background: white; padding: 12px 15px; border: 1px solid #e2e8f0; border-radius: 10px; cursor: pointer; transition: background 0.2s;">
                <input type="radio" name="gp_prior_opt" value="4" style="accent-color: #2563eb; transform: scale(1.1); cursor: pointer;">
                <span style="font-size: 0.92rem;">Set <math display="inline"><mi>&mu;</mi></math> to a large positive value and <math display="inline"><msup><mi>&mu;</mi><mo>&prime;</mo></msup></math> to a negative value</span>
            </label>
            
            <label style="display: flex; align-items: center; gap: 10px; background: white; padding: 12px 15px; border: 1px solid #e2e8f0; border-radius: 10px; cursor: pointer; transition: background 0.2s;">
                <input type="radio" name="gp_prior_opt" value="5" style="accent-color: #2563eb; transform: scale(1.1); cursor: pointer;">
                <span style="font-size: 0.92rem;">Set <math display="inline"><mi>&mu;</mi></math> to a negative value and <math display="inline"><msup><mi>&mu;</mi><mo>&prime;</mo></msup></math> to a large positive value</span>
            </label>
        </div>
        
        <button onclick="checkGpPriorAnswer()" style="background: #2563eb; color: white; border: none; padding: 10px 20px; font-size: 0.9rem; font-weight: 600; border-radius: 8px; cursor: pointer; transition: background 0.2s;">
            Submit Answer
        </button>
        
        <div id="gp_quiz_feedback" style="margin-top: 15px; display: none; padding: 12px 15px; border-radius: 10px; font-size: 0.9rem; line-height: 1.4;"></div>
        
    </div>

    <script>
    function checkGpPriorAnswer() {{
        const selected = document.querySelector('input[name="gp_prior_opt"]:checked');
        const feedbackDiv = document.getElementById('gp_quiz_feedback');
        
        if (!selected) {{
            feedbackDiv.style.display = "block";
            feedbackDiv.style.background = "#fee2e2";
            feedbackDiv.style.color = "#991b1b";
            feedbackDiv.style.border = "1px solid #fca5a5";
            feedbackDiv.innerHTML = "<strong>⚠️ Selection Missing:</strong> Please pick an option before submitting!";
            return;
        }}
        
        feedbackDiv.style.display = "block";
        if (selected.value === "4") {{
            feedbackDiv.style.background = "#dcfce7";
            feedbackDiv.style.color = "#166534";
            feedbackDiv.style.border = "1px solid #bbf7d0";
            feedbackDiv.innerHTML = "<strong>✅ Correct!</strong> This encourages exploration across the mission area while remaining conservative and highly cautious about the underlying safety constraints.";
        }} else {{
            feedbackDiv.style.background = "#f1f5f9";
            feedbackDiv.style.color = "#475569";
            feedbackDiv.style.border = "1px solid #cbd5e1";
            feedbackDiv.innerHTML = "<strong>❌ Try Again:</strong> How can you encourage exploring new points while respecting safety constraints as much as possible?";
        }}
    }}
    </script>
    """
    return HTML(html_content)




def explain_zones_definition():
    """Generates a responsive HTML/CSS dashboard panel defining the active learning 
    zones and transductive loop mechanics using native HTML5 MathML.
    """
    html_content = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
                max-width: 900px; margin: 20px auto; padding: 30px; background-color: #f8fafc; 
                border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 20px rgba(0,0,0,0.03);
                color: #334155;">
        
        <div style="text-align: center; margin-bottom: 25px;">
            <span style="background: #e0f2fe; color: #0369a1; padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
                🧭 Strategy Formulation
            </span>
            <h2 style="color: #1e293b; margin-top: 10px; margin-bottom: 12px; font-size: 1.6rem; font-weight: 800; letter-spacing: -0.025em;">
                Translating Beliefs into Decision Rules
            </h2>
        </div>

        <p style="font-size: 0.95rem; color: #475569; margin: 0 0 20px 0; line-height: 1.5; text-align: left;">
            Now comes the real question: How do you effectively translate your beliefs into decision rules? Thanks to the mathematical properties of Gaussian Processes, you can leverage uncertainty quantification by considering the following zones in your map:
        </p>
        
        <div style="display: flex; flex-direction: column; gap: 15px; margin-bottom: 25px;">
            
            <div style="display: flex; gap: 15px; background: white; padding: 18px; border-radius: 12px; border-left: 6px solid #16a34a; box-shadow: 0 2px 8px rgba(0,0,0,0.01);">
                <div style="background: #dcfce7; color: #166534; width: 42px; height: 42px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; flex-shrink: 0; font-weight: bold;">🟢</div>
                <div>
                    <h5 style="margin: 0 0 4px 0; color: #1e293b; font-size: 1.05rem; font-weight: 700; display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                        Pessimistic Safe Set 
                        <math display="inline"><msub><mi>S</mi><mi>n</mi></msub></math>
                        <span style="font-size: 0.8rem; font-weight: 500; background: #f0fdf4; color: #166534; padding: 2px 8px; border-radius: 12px; border: 1px solid #bbf7d0;">"Green Zone" / Sample Space</span>
                    </h5>
                    <p style="margin: 0; color: #64748b; font-size: 0.9rem; line-height: 1.5;">
                        According to conservative/pessimistic safety estimates, you are fairly confident that the rover will not break in this region. You will pick your next drilling site 
                        <math display="inline"><msub><mi>X</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub></math> here.
                    </p>
                </div>
            </div>
            
            <div style="display: flex; gap: 15px; background: white; padding: 18px; border-radius: 12px; border-left: 6px solid #eab308; box-shadow: 0 2px 8px rgba(0,0,0,0.01);">
                <div style="background: #fef9c3; color: #854d0e; width: 42px; height: 42px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; flex-shrink: 0; font-weight: bold;">🟡</div>
                <div>
                    <h5 style="margin: 0 0 4px 0; color: #1e293b; font-size: 1.05rem; font-weight: 700; display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                        Optimistic Safe Set 
                        <math display="inline"><msub><mover accent="true"><mi>S</mi><mo>&#x00302;</mo></mover><mi>n</mi></msub></math>
                        <span style="font-size: 0.8rem; font-weight: 500; background: #fefce8; color: #854d0e; padding: 2px 8px; border-radius: 12px; border: 1px solid #fef08a;">"Yellow Zone"</span>
                    </h5>
                    <p style="margin: 0; color: #64748b; font-size: 0.9rem; line-height: 1.5;">
                        This is the broader territory that might be safe according to your optimistic safety estimates. It bounds the region where valuable information could be unlocked (i.e. the target space below).
                    </p>
                </div>
            </div>
            
            <div style="display: flex; gap: 15px; background: white; padding: 18px; border-radius: 12px; border-left: 6px solid #d946ef; box-shadow: 0 2px 8px rgba(0,0,0,0.01);">
                <div style="background: #fdf4ff; color: #86198f; width: 42px; height: 42px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; flex-shrink: 0; font-weight: bold;">🗺️</div>
                <div>
                    <h5 style="margin: 0 0 4px 0; color: #1e293b; font-size: 1.05rem; font-weight: 700; display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                        Potential Maximizers 
                        <math display="inline"><msub><mi>A</mi><mi>n</mi></msub></math>
                        <span style="font-size: 0.8rem; font-weight: 500; background: #fdf4ff; color: #86198f; padding: 2px 8px; border-radius: 12px; border: 1px solid #f5d0fe;">"Treasure Map" / Target Space</span>
                    </h5>
                    <p style="margin: 0; color: #64748b; font-size: 0.9rem; line-height: 1.5;">
                        This is a promising subset of the Yellow Zone. A point qualifies for the Treasure Map if its optimistic mineral value estimate is higher than the highest pessimistic mineral value estimates in the Green Zone. This is where the jackpot might be hiding.
                    </p>
                </div>
            </div>
            
        </div>
        
        <div style="margin-bottom: 25px; line-height: 1.6; font-size: 0.95rem; color: #475569;">
            <p style="margin: 0 0 15px 0;">
                You want to find the jackpot inside the Treasure Map <math display="inline"><msub><mi>A</mi><mi>n</mi></msub></math>, but your rover is physically trapped inside the Green Zone <math display="inline"><msub><mi>S</mi><mi>n</mi></msub></math> which may be smaller or even lie completely outside of <math display="inline"><msub><mi>A</mi><mi>n</mi></msub></math>. How is this possible?
            </p>
            <p style="margin: 0;">
                Thankfully, researchers have developed a powerful framework called <strong>Transductive Active Learning</strong> which addresses this challenge. The guiding principle is the following:
            </p>
        </div>
        
        <div style="background: #fef2f2; border: 1px solid #fee2e2; padding: 15px 20px; border-radius: 12px; text-align: center; margin-bottom: 25px;">
            <p style="margin: 0; color: #991b1b; font-size: 0.95rem; line-height: 1.5; font-weight: 700; font-style: italic;">
                Select samples to minimize the posterior "uncertainty" about the function values within the target space.
            </p>
        </div>
        
        <div style="font-size: 0.9rem; line-height: 1.6; font-size: 0.95rem; color: #475569;">
            <p style="margin: 0 0 12px 0;">
                The algorithm will look at the Treasure Map <math display="inline"><msub><mi>A</mi><mi>n</mi></msub></math> and find a point to drill inside the safe Zone <math display="inline"><msub><mi>S</mi><mi>n</mi></msub></math> which gives us the most information about the concentration in <math display="inline"><msub><mi>A</mi><mi>n</mi></msub></math>. By drilling safely inside the boundary, the information will "leak" across the line, shrinking your uncertainty in the dangerous territory and safely unlocking the path to the reachable maximum.
            </p>
            <p style="margin: 0;">
                There are many ways to measure uncertainty. One effective measure is the entropy of the prediction targets, which gives rise to Information-Based Transductive Learning (ITL). Under some regularity assumptions, this learning algorithm will converge to the reachable maximum without violating the constraints (with high probability). You will employ this algorithm to complete this mission.
            </p>
        </div>
        
    </div>
    """
    return HTML(html_content)

def explain_confidence_quantification():
    """Generates a responsive HTML/CSS dashboard panel defining the confidence 
    intervals and full Gaussian Process equations using native HTML5 MathML.
    """
    html_content = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
                max-width: 900px; margin: 20px auto; padding: 30px; background-color: #f8fafc; 
                border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 20px rgba(0,0,0,0.03);
                color: #334155;">
        
        <div style="text-align: center; margin-bottom: 25px;">
            <span style="background: #ede9fe; color: #5b21b6; padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
                📊 Statistical Bounds
            </span>
            <h2 style="color: #1e293b; margin-top: 10px; margin-bottom: 12px; font-size: 1.6rem; font-weight: 800; letter-spacing: -0.025em;">
                Confidence Intervals & Uncertainty Quantification
            </h2>
        </div>

        <p style="font-size: 0.95rem; color: #475569; margin: 0 0 15px 0; line-height: 1.5; text-align: left;">
            By optimistic and pessimistic estimates, we refer to statistical confidence intervals calculated as 
            <math display="inline"><mo>&plusmn;</mo><mi>&beta;</mi><mo>&times;</mo></math>(standard deviation) around the predicted posterior mean. 
        </p>
        
        <p style="font-size: 0.95rem; color: #475569; margin: 0 0 20px 0; line-height: 1.5; text-align: left;">
            Specifically, we fix parameters 
            <math display="inline"><msub><mi>&beta;</mi><mi>f</mi></msub><mo>,</mo><msub><mi>&beta;</mi><mi>g</mi></msub><mo>&gt;</mo><mn>0</mn></math> 
            and define the operational confidence bounds at point <math display="inline"><mi>X</mi></math> as:
        </p>

        <div style="margin: 15px 0 25px 0; background: #f1f5f9; padding: 15px; border-radius: 10px; color: #1e293b; text-align: center; width: 100%; box-sizing: border-box;">
            <math display="block" style="margin-bottom: 10px; width: 100%;">
                <mo>[</mo><msubsup><mi>&ell;</mi><mi>n</mi><mi>f</mi></msubsup><mo>(</mo><mi>X</mi><mo>)</mo><mo>,</mo><msubsup><mi>X</mi><mi>n</mi><mi>f</mi></msubsup><mo>(</mo><mi>X</mi><mo>)</mo><mo>]</mo>
                <mo>:=</mo>
                <mo>[</mo>
                <msub><mi>&mu;</mi><mi>n</mi></msub><mo>(</mo><mi>X</mi><mo>)</mo><mo>-</mo><msub><mi>&beta;</mi><mi>f</mi></msub><msub><mi>&sigma;</mi><mi>n</mi></msub><mo>(</mo><mi>X</mi><mo>)</mo
                ><mo>,</mo>
                <msub><mi>&mu;</mi><mi>n</mi></msub><mo>(</mo><mi>X</mi><mo>)</mo><mo>+</mo><msub><mi>&beta;</mi><mi>f</mi></msub><msub><mi>&sigma;</mi><mi>n</mi></msub><mo>(</mo><mi>X</mi><mo>)</mo
                ><mo>]</mo>
            </math>
            <math display="block" style="width: 100%;">
                <mo>[</mo><msubsup><mi>&ell;</mi><mi>n</mi><mi>g</mi></msubsup><mo>(</mo><mi>X</mi><mo>)</mo><mo>,</mo><msubsup><mi>X</mi><mi>n</mi><mi>g</mi></msubsup><mo>(</mo><mi>X</mi><mo>)</mo><mo>]</mo>
                <mo>:=</mo>
                <mo>[</mo>
                <msubsup><mi>&mu;</mi><mi>n</mi><mo>&prime;</mo></msubsup><mo>(</mo><mi>X</mi><mo>)</mo><mo>-</mo><msub><mi>&beta;</mi><mi>g</mi></msub><msubsup><mi>&sigma;</mi><mi>n</mi><mo>&prime;</mo></msubsup><mo>(</mo><mi>X</mi><mo>)</mo
                ><mo>,</mo>
                <msubsup><mi>&mu;</mi><mi>n</mi><mo>&prime;</mo></msubsup><mo>(</mo><mi>X</mi><mo>)</mo><mo>+</mo><msub><mi>&beta;</mi><mi>g</mi></msub><msubsup><mi>&sigma;</mi><mi>n</mi><mo>&prime;</mo></msubsup><mo>(</mo><mi>X</mi><mo>)</mo
                ><mo>]</mo>
            </math>
        </div>

        <p style="font-size: 0.95rem; color: #475569; margin: 0 0 25px 0; line-height: 1.5; text-align: left;">
            where the localized standard deviations are evaluated using the kernels as 
            <math display="inline"><msub><mi>&sigma;</mi><mi>n</mi></msub><mo>(</mo><mi>X</mi><mo>)</mo><mo>:=</mo><msub><mi>k</mi><mi>n</mi></msub><msup><mrow><mo>(</mo><mi>X</mi><mo>,</mo><mi>X</mi><mo>)</mo></mrow><mrow><mn>1</mn><mo>/</mo><mn>2</mn></mrow></msup></math> 
            and 
            <math display="inline"><msubsup><mi>&sigma;</mi><mi>n</mi><mo>&prime;</mo></msubsup><mo>(</mo><mi>X</mi><mo>)</mo><mo>:=</mo><msubsup><mi>k</mi><mi>n</mi><mo>&prime;</mo></msubsup><msup><mrow><mo>(</mo><mi>X</mi><mo>,</mo><mi>X</mi><mo>)</mo></mrow><mrow><mn>1</mn><mo>/</mo><mn>2</mn></mrow></msup></math>.
        </p>

        <h4 style="color: #1e293b; font-size: 1.1rem; margin-top: 0; margin-bottom: 12px; font-weight: 700;">
            ⚙️ Gaussian Process Posterior Reference Equations
        </h4>
        <p style="font-size: 0.9rem; color: #64748b; margin: 0 0 15px 0; line-height: 1.5;">
            As a reminder, the formal conditional posterior mean and updated covariance structures for the objective function 
            <math display="inline"><mi>f</mi></math> and safety matrix 
            <math display="inline"><mi>g</mi></math> are dictated by the following telemetry formulas (where <math display="inline"><mi>&sigma;</mi></math> is the std of measurement noise and <math display="inline"><mi>X</mi><mo>,</mo><mi>U</mi><mo>,</mo><mi>V</mi><mo>∈</mo><mi mathvariant="bold">R</mi></math>):
        </p>

        <div style="background: white; padding: 18px; border-radius: 12px; border-left: 5px solid #3b82f6; box-shadow: 0 2px 8px rgba(0,0,0,0.01); margin-bottom: 15px;">
            <h5 style="margin: 0 0 10px 0; color: #1e293b; font-size: 0.95rem; font-weight: 700;">
                Objective Updates (<math display="inline"><mi>f</mi></math> distribution given <math display="inline"><msubsup><mi>D</mi><mi>n</mi><mi>f</mi></msubsup></math>)
            </h5>
            <div style="background: #f8fafc; padding: 12px; border-radius: 8px; overflow-x: auto;">
                <math display="block" style="width: 100%; text-align: left;">
                    <msub><mi>&mu;</mi><mi>n</mi></msub><mo>(</mo><mi>X</mi><mo>)</mo><mo>=</mo><mi>&mu;</mi><mo>(</mo><mi>X</mi><mo>)</mo><mo>+</mo><mi mathvariant="bold">k</mi><mo>(</mo><mi>X</mi><mo>,</mo><msub><mi>D</mi><mi>n</mi></msub><mo>)</mo><msup><mrow><mo>(</mo><mi mathvariant="bold">K</mi><mo>(</mo><msub><mi>D</mi><mi>n</mi></msub><mo>,</mo><msub><mi>D</mi><mi>n</mi></msub><mo>)</mo><mo>+</mo><msup><mi>&sigma;</mi><mn>2</mn></msup><mi mathvariant="bold">I</mi><mo>)</mo></mrow><mrow><mo>-</mo><mn>1</mn></mrow></msup><mo>(</mo><msub><mi mathvariant="bold">y</mi><mi>1:n</mi></msub><mo>-</mo><msub><mi>&mu;</mi><msub><mi>D</mi><mi>n</mi></msub></msub><mo>)</mo>
                </math>
                <div style="margin: 8px 0;"></div>
                <math display="block" style="width: 100%; text-align: left;">
                    <msub><mi>k</mi><mi>n</mi></msub><mo>(</mo><mi>U</mi><mo>,</mo><mi>V</mi><mo>)</mo><mo>=</mo><mi>k</mi><mo>(</mo><mi>U</mi><mo>,</mo><mi>V</mi><mo>)</mo><mo>-</mo><mi mathvariant="bold">k</mi><mo>(</mo><mi>U</mi><mo>,</mo><msub><mi>D</mi><mi>n</mi></msub><mo>)</mo><msup><mrow><mo>(</mo><mi mathvariant="bold">K</mi><mo>(</mo><msub><mi>D</mi><mi>n</mi></msub><mo>,</mo><msub><mi>D</mi><mi>n</mi></msub><mo>)</mo><mo>+</mo><msup><mi>&sigma;</mi><mn>2</mn></msup><mi mathvariant="bold">I</mi><mo>)</mo></mrow><mrow><mo>-</mo><mn>1</mn></mrow></msup><mi mathvariant="bold">k</mi><mo>(</mo><msub><mi>D</mi><mi>n</mi></msub><mo>,</mo><mi>V</mi><mo>)</mo>
                </math>
            </div>
        </div>

        <div style="background: white; padding: 18px; border-radius: 12px; border-left: 5px solid #10b981; box-shadow: 0 2px 8px rgba(0,0,0,0.01);">
            <h5 style="margin: 0 0 10px 0; color: #1e293b; font-size: 0.95rem; font-weight: 700;">
                Safety Updates (<math display="inline"><mi>g</mi></math> distribution given <math display="inline"><msubsup><mi>D</mi><mi>n</mi><mi>g</mi></msubsup></math>)
            </h5>
            <div style="background: #f8fafc; padding: 12px; border-radius: 8px; overflow-x: auto;">
                <math display="block" style="width: 100%; text-align: left;">
                    <msubsup><mi>&mu;</mi><mi>n</mi><mo>&prime;</mo></msubsup><mo>(</mo><mi>X</mi><mo>)</mo><mo>=</mo><msup><mi>&mu;</mi><mo>&prime;</mo></msup><mo>(</mo><mi>X</mi><mo>)</mo><mo>+</mo><msup><mi mathvariant="bold">k</mi><mo>&prime;</mo></msup><mo>(</mo><mi>X</mi><mo>,</mo><msub><mi>D</mi><mi>n</mi></msub><mo>)</mo><msup><mrow><mo>(</mo><msup><mi mathvariant="bold">K</mi><mo>&prime;</mo></msup><mo>(</mo><msub><mi>D</mi><mi>n</mi></msub><mo>,</mo><msub><mi>D</mi><mi>n</mi></msub><mo>)</mo><mo>+</mo><msup><mi>&sigma;</mi><mn>2</mn></msup><mi mathvariant="bold">I</mi><mo>)</mo></mrow><mrow><mo>-</mo><mn>1</mn></mrow></msup><mo>(</mo><msub><mi mathvariant="bold">z</mi><mi>1:n</mi></msub><mo>-</mo><msubsup><mi>&mu;</mi><msub><mi>D</mi><mi>n</mi></msub><mo>&prime;</mo></msubsup><mo>)</mo>
                </math>
                <div style="margin: 8px 0;"></div>
                <math display="block" style="width: 100%; text-align: left;">
                    <msubsup><mi>k</mi><mi>n</mi><mo>&prime;</mo></msubsup><mo>(</mo><mi>U</mi><mo>,</mo><mi>V</mi><mo>)</mo><mo>=</mo><msup><mi>k</mi><mo>&prime;</mo></msup><mo>(</mo><mi>U</mi><mo>,</mo><mi>V</mi><mo>)</mo><mo>-</mo><msup><mi mathvariant="bold">k</mi><mo>&prime;</mo></msup><mo>(</mo><mi>U</mi><mo>,</mo><msub><mi>D</mi><mi>n</mi></msub><mo>)</mo><msup><mrow><mo>(</mo><msup><mi mathvariant="bold">K</mi><mo>&prime;</mo></msup><mo>(</mo><msub><mi>D</mi><mi>n</mi></msub><mo>,</mo><msub><mi>D</mi><mi>n</mi></msub><mo>)</mo><mo>+</mo><msup><mi>&sigma;</mi><mn>2</mn></msup><mi mathvariant="bold">I</mi><mo>)</mo></mrow><mrow><mo>-</mo><mn>1</mn></mrow></msup><msup><mi mathvariant="bold">k</mi><mo>&prime;</mo></msup><mo>(</mo><msub><mi>D</mi><mi>n</mi></msub><mo>,</mo><mi>V</mi><mo>)</mo>
                </math>
            </div>
        </div>

    </div>
    """
    return HTML(html_content)




def explain_update_rule():
    """Generates a responsive HTML/CSS dashboard panel defining the formal mathematical
    zones and transductive active learning update rules using native HTML5 MathML.
    """
    html_content = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
                max-width: 900px; margin: 20px auto; padding: 30px; background-color: #f8fafc; 
                border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 20px rgba(0,0,0,0.03);
                color: #334155;">
        
        <div style="text-align: center; margin-bottom: 25px;">
            <span style="background: #fee2e2; color: #991b1b; padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
                🧠 Decision Engine
            </span>
            <h2 style="color: #1e293b; margin-top: 10px; margin-bottom: 12px; font-size: 1.6rem; font-weight: 800; letter-spacing: -0.025em;">
                Formal Zone Definitions & Update Rules
            </h2>
        </div>

        <p style="font-size: 0.95rem; color: #475569; margin: 0 0 15px 0; line-height: 1.5; text-align: left;">
            Now we are finally ready to formally define the operational boundaries and target zones using our statistical confidence limits:
        </p>

        <div style="margin: 15px 0 25px 0; background: #f1f5f9; padding: 18px; border-radius: 10px; color: #1e293b; text-align: left; box-sizing: border-box;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; flex-wrap: wrap; gap: 10px;">
                <math display="inline">
                    <msub><mi>S</mi><mi>n</mi></msub>
                    <mo>:=</mo>
                    <mo>{</mo><mi>X</mi><mo>&in;</mo><mi>R</mi><mo>:</mo>
                    <msubsup><mi>&ell;</mi><mi>n</mi><mi>g</mi></msubsup><mo>(</mo><mi>X</mi><mo>)</mo><mo>&ge;</mo><mn>0</mn><mo>}</mo>
                </math>
                <span style="font-size: 0.85rem; color: #64748b; font-style: italic;">(safe even by a margin of <math display="inline"><msub><mi>&beta;</mi><mi>g</mi></msub></math> standard deviations)</span>
            </div>
            
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; flex-wrap: wrap; gap: 10px;">
                <math display="inline">
                    <msub><mover accent="true"><mi>S</mi><mo>&#x00302;</mo></mover><mi>n</mi></msub>
                    <mo>:=</mo>
                    <mo>{</mo><mi>X</mi><mo>&in;</mo><mi>R</mi><mo>:</mo>
                    <msubsup><mi>u</mi><mi>n</mi><mi>g</mi></msubsup><mo>(</mo><mi>X</mi><mo>)</mo><mo>&ge;</mo><mn>0</mn><mo>}</mo>
                </math>
                <span style="font-size: 0.85rem; color: #64748b; font-style: italic;">(potentially safe when overestimating <math display="inline"><mi>g</mi></math> by <math display="inline"><msub><mi>&beta;</mi><mi>g</mi></msub></math> standard deviations)</span>
            </div>

            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <math display="inline">
                    <msub><mi>A</mi><mi>n</mi></msub>
                    <mo>:=</mo>
                    <mo>{</mo><mi>X</mi><mo>&in;</mo><msub><mover accent="true"><mi>S</mi><mo>&#x00302;</mo></mover><mi>n</mi></msub><mo>:</mo>
                    <msubsup><mi>u</mi><mi>n</mi><mi>f</mi></msubsup><mo>(</mo><mi>X</mi><mo>)</mo><mo>&ge;</mo>
                    <msub><mo>max</mo><mrow><mi>W</mi><mo>&in;</mo><msub><mi>S</mi><mi>n</mi></msub></mrow></msub>
                    <msubsup><mi>&ell;</mi><mi>n</mi><mi>f</mi></msubsup><mo>(</mo><mi>W</mi><mo>)</mo><mo>}</mo>
                </math>
                <span style="font-size: 0.85rem; color: #64748b; font-style: italic; text-align: right; max-width: 380px;">(potentially safe points whose optimistic concentration exceeds the maximum pessimistic concentration in <math display="inline"><msub><mi>S</mi><mi>n</mi></msub></math>)</span>
            </div>
        </div>

        <p style="font-size: 0.95rem; color: #475569; margin: 20px 0 15px 0; line-height: 1.5; text-align: left;">
            By continuously maintaining and upgrading our non-parametric Gaussian Process posteriors for both the mineral value function <math display="inline"><mi>f</mi></math> and the safety boundary function <math display="inline"><mi>g</mi></math>, at any stage throughout the mission, we can extract our <strong>current best prediction</strong> for the optimal accessible coordinate. This point maximizes our current point estimate while adhering to our conservative safety bounds:
        </p>

        <div style="margin: 15px 0 20px 0; background: #f1f5f9; padding: 15px; border-radius: 10px; color: #1e293b; text-align: center; width: 100%; box-sizing: border-box;">
            <math display="block" style="width: 100%;">
                <msubsup><mi>X</mi><mi>n</mi><mtext>best</mtext></msubsup>
                <mo>&in;</mo>
                <msub>
                    <mtext>argmax</mtext>
                    <mrow><mi>X</mi><mo>&in;</mo><msub><mi>S</mi><mi>n</mi></msub></mrow>
                </msub>
                <msub><mi>&mu;</mi><mi>n</mi></msub><mo>(</mo><mi>X</mi><mo>)</mo>
            </math>
        </div>

        <p style="font-size: 0.95rem; color: #475569; margin: 0 0 15px 0; line-height: 1.5; text-align: left;">
            We now define how <strong>Step 5</strong> in the ptimization workflow is executed. The explicit mathematical optimization rule for selecting your rover's next coordinate evaluation point is governed by balancing entropy reduction against physical boundaries:
        </p>

        <div style="margin: 15px 0; background: #f1f5f9; padding: 15px; border-radius: 10px; color: #1e293b; text-align: center; width: 100%; box-sizing: border-box;">
            <math display="block" style="width: 100%;">
                <msub><mi>X</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub>
                <mo>&in;</mo>
                <msub><mtext>argmin</mtext><mrow><mi>X</mi><mo>&in;</mo><msub><mi>S</mi><mi>n</mi></msub></mrow></msub>
                <mi>H</mi>
                <mo>(</mo>
                <msub><mi mathvariant="bold">f</mi><msub><mi>A</mi><mi>n</mi></msub></msub>
                <mo>|</mo>
                <msubsup><mi>D</mi><mi>n</mi><mi>f</mi></msubsup>
                <mo>,</mo>
                <msub><mi>y</mi><mi>X</mi></msub>
                <mo>)</mo>
                <mo>=</mo>
                <msub><mtext>argmax</mtext><mrow><mi>X</mi><mo>&in;</mo><msub><mi>S</mi><mi>n</mi></msub></mrow></msub>
                <mi>I</mi>
                <mo>(</mo>
                <msub><mi mathvariant="bold">f</mi><msub><mi>A</mi><mi>n</mi></msub></msub>
                <mo>;</mo>
                <msub><mi>y</mi><mi>X</mi></msub>
                <mo>|</mo>
                <msubsup><mi>D</mi><mi>n</mi><mi>f</mi></msubsup>
                <mo>)</mo>
            </math>
        </div>

        <p style="font-size: 0.95rem; color: #475569; margin: 0 0 20px 0; line-height: 1.5; text-align: left;">
            where <math display="inline"><mi>H</mi></math> is the (conditional) differential entropy and <math display="inline"><mi>I</mi></math> is the (conditional) information gain. Intuitively, the formula says:
        </p>

        <div style="background: #fef2f2; border: 1px solid #fee2e2; padding: 15px 20px; border-radius: 12px; text-align: center; margin-bottom: 25px;">
            <p style="margin: 0; color: #991b1b; font-size: 0.95rem; line-height: 1.5; font-weight: 700; font-style: italic;">
                Pick the point X in the sample space <math display="inline"><msub><mi>S</mi><mi>n</mi></msub></math> leading to the maximum uncertainty reduction for <math display="inline"><mi>f</mi></math> on our target space <math display="inline"><msub><mi>A</mi><mi>n</mi></msub></math>.
            </p>
        </div>

        <p style="font-size: 0.95rem; color: #475569; margin: 0 0 15px 0; line-height: 1.5; text-align: left;">
            In our Gaussian Process setting, things are even simpler because the objective can be expressed as follows:
        </p>

        <div style="margin: 15px 0 15px 0; background: #f1f5f9; padding: 15px; border-radius: 10px; color: #1e293b; text-align: center; width: 100%; box-sizing: border-box;">
            <math display="block" style="width: 100%;">
                <mi>I</mi>
                <mo>(</mo>
                <msub><mi mathvariant="bold">f</mi><msub><mi>A</mi><mi>n</mi></msub></msub>
                <mo>;</mo>
                <msub><mi>y</mi><mi>X</mi></msub>
                <mo>|</mo>
                <msubsup><mi>D</mi><mi>n</mi><mi>f</mi></msubsup>
                <mo>)</mo>
                <mo>=</mo>
                <mfrac><mn>1</mn><mn>2</mn></mfrac>
                <mspace width="0.2em"></mspace>
                <mi>log</mi>
                <mspace width="0.2em"></mspace>
                <mfrac>
                    <mrow><mtext>Var</mtext><mo>[</mo><msub><mi>y</mi><mi>X</mi></msub><mo>|</mo><msubsup><mi>D</mi><mi>n</mi><mi>f</mi></msubsup><mo>]</mo></mrow>
                    <mrow><mtext>Var</mtext><mo>[</mo><msub><mi>y</mi><mi>X</mi></msub><mo>|</mo><msub><mi mathvariant="bold">f</mi><msub><mi>A</mi><mi>n</mi></msub></msub><mo>,</mo><msubsup><mi>D</mi><mi>n</mi><mi>f</mi></msubsup><mo>]</mo></mrow>
                </mfrac>
                <mo>=</mo>
                <mfrac><mn>1</mn><mn>2</mn></mfrac>
                <mspace width="0.2em"></mspace>
                <mi>log</mi>
                <mspace width="0.2em"></mspace>
                <mfrac>
                    <mrow><msub><mi>k</mi><mi>n</mi></msub><mo>(</mo><mi>X</mi><mo>,</mo><mi>X</mi><mo>)</mo><mo>+</mo><msup><mi>&sigma;</mi><mn>2</mn></msup></mrow>
                    <mrow><msub><mover><mi>k</mi><mo>~</mo></mover><mi>n</mi></msub><mo>(</mo><mi>X</mi><mo>,</mo><mi>X</mi><mo>)</mo><mo>+</mo><msup><mi>&sigma;</mi><mn>2</mn></msup></mrow>
                </mfrac>
            </math>
        </div>

        <p style="font-size: 0.95rem; color: #475569; margin: 0 0 15px 0; line-height: 1.5; text-align: left;">
            where the conditional variance in the denominator is given by
        </p>

        <div style="margin: 15px 0 25px 0; background: #f1f5f9; padding: 15px; border-radius: 10px; color: #1e293b; text-align: center; width: 100%; box-sizing: border-box;">
            <math display="block" style="width: 100%;">
                <msub><mover><mi>k</mi><mo>~</mo></mover><mi>n</mi></msub><mo>(</mo><mi>U</mi><mo>,</mo><mi>U</mi><mo>)</mo>
                <mo>=</mo>
                <msub><mi>k</mi><mi>n</mi></msub><mo>(</mo><mi>U</mi><mo>,</mo><mi>U</mi><mo>)</mo>
                <mo>-</mo>
                <msub><mi mathvariant="bold">k</mi><mi>n</mi></msub><mo>(</mo><mi>U</mi><mo>,</mo><msub><mi>A</mi><mi>n</mi></msub><mo>)</mo>
                <msup>
                    <mrow>
                        <msub><mi mathvariant="bold">K</mi><mi>n</mi></msub>
                        <mo>(</mo><msub><mi>A</mi><mi>n</mi></msub><mo>,</mo><msub><mi>A</mi><mi>n</mi></msub><mo>)</mo>
                    </mrow>
                    <mrow><mo>-</mo><mn>1</mn></mrow>
                </msup>
                <msub><mi mathvariant="bold">k</mi><mi>n</mi></msub><mo>(</mo><msub><mi>A</mi><mi>n</mi></msub><mo>,</mo><mi>U</mi><mo>)</mo>
            </math>
        </div>

        <h4 style="color: #1e293b; font-size: 1.1rem; margin-top: 0; margin-bottom: 12px; font-weight: 700;">
            🏁 Mission Ready: Commencing Execution
        </h4>
        <div style="font-size: 0.95rem; line-height: 1.6; color: #475569;">
            <p style="margin: 0 0 12px 0;">
                By combining Gaussian Process regressions with Information-Based Transductive Learning (ITL), you have developed conceptual understanding of how to design a risk-aware optimization loop. The rover does not blindly chase maximum parameters in unverified terrain, nor does it get stuck in local safety clusters. 
            </p>
            <p style="margin: 0; background: #f0fdf4; border: 1px solid #bbf7d0; padding: 15px; border-radius: 12px; color: #166534; font-weight: 500;">
                Now that the framework is established, your mathematical toolkit is locked and ready. Proceed to the next block cells to complete the implementation and begin mapping the Martian mineral veins safely!
            </p>
        </div>

    </div>
    """
    return HTML(html_content)


def gp_hyperparameters_quiz():
    """Generates an interactive, responsive checkbox quiz about the effects
    of Beta scaling parameters on Safe Bayesian Optimization zones and risks.
    """
    uid = "quiz_gp_beta_tradeoffs"
    
    html_content = f"""
    <div id="{uid}" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
                max-width: 900px; margin: 25px auto; padding: 25px; background-color: #f8fafc; 
                border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 20px rgba(0,0,0,0.03); color: #334155;">
        
        <div style="margin-bottom: 20px;">
            <span style="background: #ede9fe; color: #5b21b6; padding: 5px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
                🧠 Parameter Tradeoffs Check
            </span>
            <h4 style="color: #1e293b; margin-top: 10px; margin-bottom: 8px; font-size: 1.1rem; font-weight: 700; line-height: 1.4;">
                Assuming all other variables are held completely fixed, which of the following statements regarding the exploration parameters 
                <math display="inline"><msub><mi>&beta;</mi><mi>f</mi></msub></math> and 
                <math display="inline"><msub><mi>&beta;</mi><mi>g</mi></msub></math> are correct?
            </h4>
            <p style="font-size: 0.85rem; color: #64748b; margin: 0;"><em>Select ALL that apply (Multiple choices may be correct):</em></p>
        </div>
        
        <div style="display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px;">
            
            <label style="display: flex; align-items: flex-start; gap: 12px; background: white; padding: 12px 15px; border: 1px solid #e2e8f0; border-radius: 10px; cursor: pointer; transition: background 0.2s;">
                <input type="checkbox" name="gp_beta_opt" value="1" style="accent-color: #2563eb; transform: scale(1.1); margin-top: 3px; cursor: pointer;">
                <span style="font-size: 0.92rem;">A smaller safety parameter <math display="inline"><msub><mi>&beta;</mi><mi>g</mi></msub></math> results in a larger, less conservative Safe Set <math display="inline"><msub><mi>S</mi><mi>n</mi></msub></math>.</span>
            </label>
            
            <label style="display: flex; align-items: flex-start; gap: 12px; background: white; padding: 12px 15px; border: 1px solid #e2e8f0; border-radius: 10px; cursor: pointer; transition: background 0.2s;">
                <input type="checkbox" name="gp_beta_opt" value="2" style="accent-color: #2563eb; transform: scale(1.1); margin-top: 3px; cursor: pointer;">
                <span style="font-size: 0.92rem;">A smaller safety parameter <math display="inline"><msub><mi>&beta;</mi><mi>g</mi></msub></math> shrinks the Optimistic Safe Set <math display="inline"><msub><mover accent="true"><mi>S</mi><mo>&#x00302;</mo></mover><mi>n</mi></msub></math>.</span>
            </label>
            
            <label style="display: flex; align-items: flex-start; gap: 12px; background: white; padding: 12px 15px; border: 1px solid #e2e8f0; border-radius: 10px; cursor: pointer; transition: background 0.2s;">
                <input type="checkbox" name="gp_beta_opt" value="3" style="accent-color: #2563eb; transform: scale(1.1); margin-top: 3px; cursor: pointer;">
                <span style="font-size: 0.92rem;">Decreasing the value of <math display="inline"><msub><mi>&beta;</mi><mi>g</mi></msub></math> exposes the rover to a higher operational risk of crashing.</span>
            </label>
            
            <label style="display: flex; align-items: flex-start; gap: 12px; background: white; padding: 12px 15px; border: 1px solid #e2e8f0; border-radius: 10px; cursor: pointer; transition: background 0.2s;">
                <input type="checkbox" name="gp_beta_opt" value="4" style="accent-color: #2563eb; transform: scale(1.1); margin-top: 3px; cursor: pointer;">
                <span style="font-size: 0.92rem;">A smaller mineral parameter <math display="inline"><msub><mi>&beta;</mi><mi>f</mi></msub></math> narrows the search space by shrinking the Potential Maximizers set <math display="inline"><msub><mi>A</mi><mi>n</mi></msub></math>.</span>
            </label>
            
            <label style="display: flex; align-items: flex-start; gap: 12px; background: white; padding: 12px 15px; border: 1px solid #e2e8f0; border-radius: 10px; cursor: pointer; transition: background 0.2s;">
                <input type="checkbox" name="gp_beta_opt" value="5" style="accent-color: #2563eb; transform: scale(1.1); margin-top: 3px; cursor: pointer;">
                <span style="font-size: 0.92rem;">Larger overall <math display="inline"><mi>&beta;</mi></math> values expand confidence bounds, actively encouraging wider exploration of unknown spaces.</span>
            </label>
            
            <label style="display: flex; align-items: flex-start; gap: 12px; background: white; padding: 12px 15px; border: 1px solid #e2e8f0; border-radius: 10px; cursor: pointer; transition: background 0.2s;">
                <input type="checkbox" name="gp_beta_opt" value="6" style="accent-color: #2563eb; transform: scale(1.1); margin-top: 3px; cursor: pointer;">
                <span style="font-size: 0.92rem;">Setting smaller <math display="inline"><mi>&beta;</mi></math> values means relying more heavily on the model's current mean beliefs (exploitation).</span>
            </label>
        </div>
        
        <button onclick="checkBetaCheckboxQuiz()" style="background: #2563eb; color: white; border: none; padding: 10px 20px; font-size: 0.9rem; font-weight: 600; border-radius: 8px; cursor: pointer; transition: background 0.2s;">
            Submit Answers
        </button>
        
        <div id="beta_quiz_feedback" style="margin-top: 15px; display: none; padding: 12px 15px; border-radius: 10px; font-size: 0.9rem; line-height: 1.4;"></div>
        
    </div>

    <script>
    function checkBetaCheckboxQuiz() {{
        const checkedBoxes = document.querySelectorAll('input[name="gp_beta_opt"]:checked');
        const feedbackDiv = document.getElementById('beta_quiz_feedback');
        
        if (checkedBoxes.length === 0) {{
            feedbackDiv.style.display = "block";
            feedbackDiv.style.background = "#fee2e2";
            feedbackDiv.style.color = "#991b1b";
            feedbackDiv.style.border = "1px solid #fca5a5";
            feedbackDiv.innerHTML = "<strong>⚠️ Selection Missing:</strong> Please select at least one statement before evaluating!";
            return;
        }}
        
        // Map checked values to an array of integers
        const chosenValues = Array.from(checkedBoxes).map(cb => parseInt(cb.value));
        
        // All 6 options are scientifically correct statements based on the definitions
        const totalCorrectNeeded = 6;
        const isCompletelyCorrect = chosenValues.length === totalCorrectNeeded;

        feedbackDiv.style.display = "block";
        if (isCompletelyCorrect) {{
            feedbackDiv.style.background = "#dcfce7";
            feedbackDiv.style.color = "#166534";
            feedbackDiv.style.border = "1px solid #bbf7d0";
            feedbackDiv.innerHTML = "<strong>🎉 Excellent! All selected statements are correct.</strong> You have perfectly mastered how these scaling parameters manage the balance between physical safety constraints and math exploration values.";
        }} else {{
            feedbackDiv.style.background = "#fffbeb";
            feedbackDiv.style.color = "#92400e";
            feedbackDiv.style.border = "1px solid #fde68a";
            feedbackDiv.innerHTML = "<strong>⚠️ Partially Correct:</strong> Your selected options are true, but you haven't uncovered all of them! Consider the math definitions carefully—how does changing the uncertainty weight affect both the upper and lower confidence thresholds? Try checking the remaining options.";
        }}
    }}
    </script>
    """
    return HTML(html_content)

# GP-related
def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return variance * np.exp(-0.5 * sqdist / length_scale**2)
def compute_posterior_from_observations(gp_obj):
    """Executes the standard Gaussian Process conditioning equations.
    Moves complex linear algebra routines outside the student-facing class.
    """
    # K_AX = K(X_train, X_grid)
    K_AX = gp_obj.kernel_func(gp_obj.X_train, gp_obj.X_grid)
    
    # K_AAn = K(X_train, X_train) + noise_var * I
    K_AAn = gp_obj.kernel_func(gp_obj.X_train, gp_obj.X_train) \
            + gp_obj.noise_var * np.eye(gp_obj.X_train.shape[0])

    # Compute: alpha = K_AAn^-1 @ (y_train - prior_mean)
    alpha = solve(K_AAn, gp_obj.y_train - gp_obj.prior_mean)
    assert not np.isnan(alpha).any(), "Got NaN values in matrix inversion. Enforce positive definiteness."
    posterior_mean = gp_obj.mu_grid + np.dot(K_AX.T, alpha)

    # Compute posterior covariance: K_grid - K_AX.T @ (K_AAn^-1 @ K_AX)
    v = solve(K_AAn, K_AX)
    assert not np.isnan(v).any(), "Got NaN values in matrix inversion. Enforce positive definiteness."
    posterior_cov = gp_obj.K_grid - np.dot(K_AX.T, v)

    return posterior_mean, posterior_cov

# Contour plotting
import matplotlib.patches as mpatches
def _plot_contour(ax, mask, edge_color, hatch_pattern, name, grid_range, extent):
    '''Plot the contour of a boolean mask on the given axis.
    Parameters:
        ax (matplotlib.axes.Axes): The axis to plot on.
        mask (np.ndarray): Boolean mask of shape (GRID_RES, GRID_RES).
        edge_color (str): Color of the contour edges.
        hatch_pattern (str): Pattern for hatching the contour
        name (str): Label for the legend.
    '''

    contours = ax.contourf(grid_range, grid_range, mask, levels=[0.5, 1.5], colors='none', hatches=[hatch_pattern])
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    contours.set_edgecolor(edge_color)
    hatch_patch = mpatches.Patch(hatch=hatch_pattern, edgecolor=edge_color, fill=False, label=name)
    return hatch_patch

def find_maxima(mission_obj):
    """Finds the maximum value of f in the safe set S_n and returns the corresponding point."""
    # Reachable max of f_star in S^* (unknown to the rover)
    mission_obj.X_STAR = GRID_COORDS[np.argmax((TRUE_F_VALS * (TRUE_G_VALS>=0)))]
    # Global max of f_star (unknown to the rover)
    mission_obj.X_STAR_GLOBAL = GRID_COORDS[np.argmax(TRUE_F_VALS)]

# Mission-related
def reset_ui_data(mission_obj):
    """Resets the mission object's internal state and UI properties to their initial conditions.
    This function is called inside the reset_data_state method."""
    mission_obj.ci_f_history = []  # confidence interval history for f
    mission_obj.ci_g_history = []  # confidence interval history for g
    mission_obj.current_best_pt = mission_obj.x_init  # best point found so far (higest mu_f)
    mission_obj.current_best_f = -np.inf # best f value found so far (higest mu_f)
    mission_obj.high_score = True  # whether new high score was achieved in the last round
    mission_obj.is_failed = False  # whether mission has failed due to constraint violation
    mission_obj.btn_step.disabled = False # for UI control
    mission_obj.result_table = []  # markdown string for the results table
    find_maxima(mission_obj)  # find the global and reachable maxima of f_star for reference

def record_confidence_intervals(mission_obj):
    """Records the current confidence intervals for both f and g at newly evaluation point.
    This function is called after each drilling step to maintain a history of confidence intervals.
    """
    x_next_idx = find_point_index(mission_obj.x_next)
    m_f, s_f = float(mission_obj.mu_f_post[x_next_idx]), float(np.sqrt(mission_obj.cov_f_post[x_next_idx, x_next_idx]))
    m_g, s_g = float(mission_obj.mu_g_post[x_next_idx]), float(np.sqrt(mission_obj.cov_g_post[x_next_idx, x_next_idx]))
    mission_obj.ci_f_history.append(f"[{m_f - mission_obj.beta_f*s_f:.3f}, {m_f + mission_obj.beta_f*s_f:.3f}]")
    mission_obj.ci_g_history.append(f"[{m_g - mission_obj.beta_g*s_g:.3f}, {m_g + mission_obj.beta_g*s_g:.3f}]")

def check_constraint_violation(mission_obj):
    true_g_val = float(g_star(mission_obj.x_next[None,:])[0])
    if true_g_val < 0:
        mission_obj.is_failed = True
        mission_obj.btn_step.disabled = True
def update_best_point(mission_obj):
    """Updates the mission object's current best point and value based on the latest posterior mean of f.
    This function is called after posterior computation to track the best known point in the safe set.
    """
    best_pt_this_round = GRID_COORDS[np.argmax(mission_obj.mu_f_post * mission_obj.S_n.flatten().astype(float))]
    true_f_val = float(f_star(best_pt_this_round[None,:])[0])
    if true_f_val > mission_obj.current_best_f:
        mission_obj.current_best_pt = best_pt_this_round
        mission_obj.current_best_f = true_f_val
        mission_obj.high_score = True
    else:
        mission_obj.high_score = False

def initialize_mission_ui(mission_obj):
    """Constructs and injects UI properties directly into the mission object."""
    # Build core widgets
    mission_obj.btn_step = widgets.Button(description="Drill Next Step 🚀", button_style="primary")
    mission_obj.btn_reset = widgets.Button(description="Reset Mission 🔄", button_style="danger")
    mission_obj.output_plots = widgets.Output()
    mission_obj.output_table = widgets.Output()
    
    # Bind UI internal callbacks using bridge adapters
    mission_obj.btn_step.on_click(lambda b: handle_step_click(mission_obj))
    mission_obj.btn_reset.on_click(lambda b: handle_reset_click(mission_obj))

def display_mission_ui(mission_obj):
    """Assembles and displays the dual-column grid layout configuration."""
    controls = widgets.HBox([mission_obj.btn_step, mission_obj.btn_reset])
    right_column = widgets.VBox([controls, mission_obj.output_table], 
                                layout=widgets.Layout(gap='10px'))
    
    main_layout = widgets.HBox(
        [mission_obj.output_plots, right_column], 
        layout=widgets.Layout(align_items='flex-start', gap='25px')
    )
    display(main_layout)
    
    # Initial render of data state
    render_mission_dashboard(mission_obj)

def handle_step_click(mission_obj):
    """Callback wrapper for step operations."""
    if mission_obj.is_failed:
        mission_obj.btn_step.disabled = True
        return
    mission_obj.conduct_drilling_round()

def handle_reset_click(mission_obj):
    """Callback wrapper for operational resetting sequences."""
    with mission_obj.output_plots:
        clear_output(wait=True)
    with mission_obj.output_table:
        clear_output(wait=True)
        
    mission_obj.reset_data_state()
    mission_obj.current_best_pt = mission_obj.x_init
    mission_obj.current_best_f = -float('inf')
    
    render_mission_dashboard(mission_obj)

def render_mission_dashboard(mission_obj):
    """Externalized rendering core for the Mission telemetry dashboard.
    Converts matplotlib streams and tracking logs into an integrated, 
    highly polished HTML5 workspace dashboard element.
    """
    # ==========================================
    # 1. RENDER MATPLOTLIB PLOTS (BASE64)
    # ==========================================
    fig = plt.figure(figsize=(9, 5.4), dpi=110) 
    gs = fig.add_gridspec(2, 2, width_ratios=[20, 20], height_ratios=[30, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    
    # Access globals/attributes carefully (assuming they are stored or imported)
    plot_contour = lambda ax, mask, edge_color, hatch_pattern, name: _plot_contour(ax, mask, edge_color, hatch_pattern, name, GRID_RANGE, EXTENT)
    vmin, vmax = 0, TRUE_F_VALS.max() + 1
    # --- Panel 1: Rover Belief Map ---
    _ = ax1.imshow(unflatten(mission_obj.mu_f_post), origin='lower', extent=EXTENT, cmap='coolwarm', vmin=vmin, vmax=vmax)   
    h1 = plot_contour(ax1, mission_obj.S_hat_n, edge_color='gray', hatch_pattern='..', name=r'$\widehat{S}_n$')
    h2 = plot_contour(ax1, mission_obj.A_n, edge_color='gold', hatch_pattern='\\\\', name=r'$A_n$')
    h3 = plot_contour(ax1, mission_obj.S_n, edge_color='green', hatch_pattern='//', name=r'$S_n$')
    elements = [h3, h1, h2]
    ax1.set_title("Belief Landscape (Color = Post. Mean of f)", fontsize=11, color='#475569', weight='600', pad=10)

    # --- Panel 2: Ground Truth Map ---
    im2 = ax2.imshow(TRUE_F_VALS, origin='lower', extent=EXTENT, cmap='coolwarm', vmin=vmin, vmax=vmax)
    h_true = plot_contour(ax2, TRUE_G_VALS >= 0, edge_color='darkseagreen', hatch_pattern='o', name=r'$S^*$')
    elements.append(h_true)
    ax2.set_title("Ground Truth Landscape (Color = f*)", fontsize=11, color='#475569', weight='600', pad=10)
    
    # Plot tracking arrays
    X_arr = np.array(mission_obj.X_history)
    h_hist, h_curr = None, None
    if len(X_arr) > 0:
        for ax in (ax1, ax2):
            if len(X_arr) > 1:
                h_hist = ax.scatter(X_arr[:-1, 0], X_arr[:-1, 1], color='white', edgecolor='#1e293b', s=55, linewidth=1.2, zorder=5, label="Past Samples")
            h_curr = ax.scatter(X_arr[-1, 0], X_arr[-1, 1], color='#f97316', edgecolor='#1e293b', s=65, linewidth=1.5, zorder=6, label="Current Sample")
    
    # Plot next point, reachable maximizer, and current best based on posteriors
    for ax in (ax1, ax2):
        h_star = ax.scatter(mission_obj.X_STAR[0], mission_obj.X_STAR[1], color='#eab308', edgecolor='#1e293b', marker='*', s=180, linewidth=1.2, zorder=7, label="Reachable Maximizer")
        h_next = ax.scatter(mission_obj.x_next[0], mission_obj.x_next[1], color='#d946ef', edgecolor='#1e293b', marker='X', s=140, linewidth=1.2, zorder=7, label="Next Sample")
        h_best = ax.scatter(mission_obj.current_best_pt[0], mission_obj.current_best_pt[1], color='#14b8a6', edgecolor='#1e293b', marker='P', s=140, linewidth=1.2, zorder=7, label=f"Current Best: $\\text{{argmax}}_{{X \in S_n}}$ $\mu_n(X)$")
        # Clean up axes style to blend with dashboard
        ax.set_aspect('equal')
        ax.tick_params(colors='#64748b', labelsize=9)
        ax.set_facecolor('#f8fafc')
        for spine in ax.spines.values():
            spine.set_color('#e2e8f0')

    elements.extend([h for h in [h_star, h_next, h_curr, h_hist, h_best] if h is not None])
    labels = [h.get_label() for h in elements]
    
    # Shared horizontal colorbar
    cax2 = fig.add_subplot(gs[1, :2])
    cb = fig.colorbar(im2, cax=cax2, orientation='horizontal')
    cb.set_label("Mineral Concentration Value", color='#64748b', fontsize=10, labelpad=5)
    cb.ax.tick_params(labelsize=9, colors='#64748b')
    cb.outline.set_color('#e2e8f0')
    
    # Legend settings
    fig.legend(elements, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=True, facecolor='#f8fafc', edgecolor='#e2e8f0', fontsize=9)
    plt.tight_layout()
    
    # Save chart to memory bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)

    # ==========================================
    # 2. COMPILE LIVE LOG ROWS
    # ==========================================
    f_star_reachable = f_star(mission_obj.X_STAR[None, :])[0]
    f_star_global = f_star(mission_obj.X_STAR_GLOBAL[None, :])[0]
    
    
    # Process history data to match table records cleanly
    if mission_obj.round > 0 and len(mission_obj.X_history) == mission_obj.round:
        pt = mission_obj.X_history[-1]
        obs_f, obs_g = mission_obj.y_history[-1], mission_obj.z_history[-1]
        ci_f_str, ci_g_str = mission_obj.ci_f_history[-1], mission_obj.ci_g_history[-1]
        idx = find_point_index(pt)
        true_f = TRUE_F_VALS.flatten()[idx]
        true_g = TRUE_G_VALS.flatten()[idx]
        
        status_badge = '<span style="background:#fee2e2; color:#991b1b; border-radius:12px; font-size:0.55rem;">CRASHED</span>' if mission_obj.is_failed else '<span style="background:#dcfce7; color:#166534; padding:2px 8px; border-radius:12px; font-size:0.55rem;">SAFE</span>'
        score_badge = '<span style="background:#fef9c3; color:#854d0e; border-radius:12px; font-size:0.55rem;">NEW BEST</span>' if mission_obj.high_score else '<span style="color:#94a3b8;">—</span>'
        
        new_row = f"""
        <tr style="border-bottom: 1px solid #f1f5f9; hover {{ background-color: #f8fafc; }}">
            <td style="padding: 10px; text-align: center; font-size: 0.6rem; color: #0f172a;">{mission_obj.round}</td>
            <td style="padding: 10px; text-align: center; font-size: 0.6rem; color: #0f172a;">({pt[0]:.1f}, {pt[1]:.1f})</td>
            <td style="padding: 10px; text-align: center; font-size: 0.6rem; color: #0f172a;">{true_f:.3f}</td>
            <td style="padding: 10px; text-align: center; font-size: 0.6rem; color: #0f172a;">{obs_f:.3f}</td>
            <td style="padding: 10px; text-align: center; font-size: 0.6rem; color: #0f172a;">{ci_f_str}</td>
            <td style="padding: 10px; text-align: center; font-size: 0.6rem; color: #0f172a;">{true_g:.3f}</td>
            <td style="padding: 10px; text-align: center; font-size: 0.6rem; color: #0f172a;">{obs_g:.3f}</td>
            <td style="padding: 10px; text-align: center; font-size: 0.6rem; color: #0f172a;">{ci_g_str}</td>
            <td style="padding: 10px; text-align: center; font-size: 0.6rem; color: #0f172a;">{status_badge}</td>
            <td style="padding: 10px; text-align: center; font-size: 0.6rem; color: #0f172a;">{score_badge}</td>
        </tr>
        """
        if not hasattr(mission_obj, 'html_rows_list'):
            mission_obj.html_rows_list = []
        mission_obj.html_rows_list.append(new_row)

    table_rows_html = "".join(getattr(mission_obj, 'html_rows_list', [])[::-1])

    # ==========================================
    # 3. STRUCTURE THE SEPARATED CODE HTML BOXES
    # ==========================================
    # HTML Left Box (Assigned completely to output_plots)
    left_plots_html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                padding: 15px; background-color: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.02);">
        <div style="background: #f8fafc; padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0; text-align: center;">
            <img src="data:image/png;base64,{img_base64}" style="max-width: 100%; height: auto; display: block; margin: 0 auto; border-radius: 6px;" alt="Mission Maps"/>
        </div>
    </div>
    """

    # HTML Right Box (Assigned completely to output_table)
    right_table_html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                color: #334155; padding: 15px; background-color: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.02);">
        
        <div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 2px solid #f1f5f9; padding-bottom: 10px; margin-bottom: 15px;">
            <div>
                <span style="background: #eff6ff; color: #1d4ed8; padding: 3px 8px; border-radius: 20px; font-size: 0.6rem; font-weight: 500; text-transform: uppercase;">📡 Telemetry Feed</span>
            </div>
            <div style="text-align: right; font-size: 0.7em; color: #64748b; line-height: 1.3;">
                <strong>Global Max:</strong> <span style="color: #1d4ed8; font-weight: 500;">{f_star_global:.3f}</span> | 
                <strong>Reachable Peak:</strong> <span style="color: #059669; font-weight: 500;">{f_star_reachable:.3f}</span>
            </div>
        </div>

        <div style="display: flex; gap: 10px; margin-bottom: 15px;">
            <div style="flex: 1; background: #f0fdf4; border: 1px solid #bbf7d0; padding: 8px 12px; border-radius: 8px;">
                <div style="font-size: 0.4rem; font-weight: 700; color: #166534; text-transform: uppercase;">Total Rounds</div>
                <div style="font-size: 0.8rem; font-weight: 800; color: #14532d;">{mission_obj.round}</div>
            </div>
            <div style="flex: 2; background: #eff6ff; border: 1px solid #bfdbfe; padding: 8px 12px; border-radius: 8px;">
                <div style="font-size: 0.4rem; font-weight: 700; color: #1e40af; text-transform: uppercase;">Highest Score</div>
                <div style="font-size: 0.8rem; font-weight: 800; color: #1e3a8a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    {mission_obj.current_best_f:.3f} <span style="font-size:0.6rem; font-weight:normal; color:#64748b;">at ({mission_obj.current_best_pt[0]:.1f}, {mission_obj.current_best_pt[1]:.1f})</span>
                </div>
            </div>
        </div>

        <div style="overflow-x: auto; border: 1px solid #e2e8f0; border-radius: 8px;">
            <table style="width: 100%; border-collapse: collapse; font-size: 0.7rem; text-align: left;">
                <thead>
                    <tr style="background-color: #f8fafc; border-bottom: 2px solid #e2e8f0; color: #475569; font-weight: 300;">
                        <th style="padding: 8px 4px; text-align: center;">Rnd</th>
                        <th style="padding: 8px 4px; text-align: center;">X</th>
                        <th style="padding: 8px 4px; text-align: center;">f*(X)</th>
                        <th style="padding: 8px 4px; text-align: center;">y</th>
                        <th style="padding: 8px 4px; text-align: center;">f(X) CI</th>
                        <th style="padding: 8px 4px; text-align: center;">g*(X)</th>
                        <th style="padding: 8px 4px; text-align: center;">z</th>
                        <th style="padding: 8px 4px; text-align: center;">g(X) CI</th>
                        <th style="padding: 8px 4px; text-align: center;">Status</th>
                        <th style="padding: 8px 4px; text-align: center;">Record</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows_html if table_rows_html else '<tr><td colspan="10" style="padding:15px; text-align:center; color:#94a3b8; font-style:italic;">Awaiting initialization sequence...</td></tr>'}
                </tbody>
            </table>
        </div>
    </div>
    """
    
    # ==========================================
    # 4. DIRECT THE LOGS SEPARATELY TO THE OBJECTS
    # ==========================================
    with mission_obj.output_plots:
        clear_output(wait=True)
        display(HTML(left_plots_html))
        
    with mission_obj.output_table:
        clear_output(wait=True)
        display(HTML(right_table_html))