from src.evaluation.decision_boundary import DecisionBoundaryCreator
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def decision_boundary_plot(model, original_model, dataloader_retrain, dataloader_train, X_forget, results_dir, plot_title=None, filename=None):
    if filename is None:
        filename = "decision_boundary"
    save_path = os.path.join(results_dir, filename) + '.pdf'

    # Set common plot styling
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Get the classes of the rogue points
    X, y = dataloader_train.dataset.X, dataloader_train.dataset.y
    X_retain, y_retain = dataloader_retrain.dataset.X, dataloader_retrain.dataset.y
    
    # If y is onehot, convert it to class indices
    if len(y.shape) == 2 and y.shape[1] > 1:
        y = torch.argmax(y, dim=1)
        y_retain = torch.argmax(y_retain, dim=1)
    
    # Professional color palette
    professional_colors = ['#ffa600', '#a05195', '#f95d6a', '#8172B3', '#CCB974', '#64B5CD']
    
    # Find the classes of the rogue points
    rogue_classes = []
    rogue_colors = []
    for i in range(X_forget.shape[0]):
        distances = np.sum((X.numpy() - X_forget[i].numpy())**2, axis=1)
        rogue_class = y[np.argmin(distances)].item()
        rogue_classes.append(rogue_class)
        rogue_colors.append(professional_colors[rogue_class % len(professional_colors)])
    
    # Create creators for both models
    unlearned_creator = DecisionBoundaryCreator(model, dataloader_retrain)
    original_creator = DecisionBoundaryCreator(original_model, dataloader_train)
    
    # Get the decision boundaries but don't plot them yet
    xx_o, yy_o, decision_boundary_original = original_creator.create_decision_boundary((-9, 9), (-9, 9))
    xx_u, yy_u, decision_boundary_unlearned = unlearned_creator.create_decision_boundary((-9, 9), (-9, 9))
    
    # Create a single figure
    fig = plt.figure(figsize=(8, 7), dpi=200, constrained_layout=True)
    
    # Get number of unique classes
    classes = np.unique(y)
    num_classes = len(classes)
    
    # Professional color palette for classes
    colors = [professional_colors[i % len(professional_colors)] for i in range(len(classes))]
    colors = [plt.matplotlib.colors.to_rgba(color) for color in colors]
    custom_cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    # Plot the unlearned decision boundary with lower opacity
    plt.pcolormesh(xx_u.numpy(), yy_u.numpy(), decision_boundary_unlearned.numpy(),
                 alpha=0.4, cmap=custom_cmap, rasterized=True)
    

    unlearned_creator.plot_retain_data(X_retain, y_retain, classes, colors)


    for class_idx in range(num_classes):
        for neighbor_class in range(class_idx + 1, num_classes):
            # Original model boundaries - dashed lines
            original_mask = np.logical_or(
                decision_boundary_original.numpy() == class_idx,
                decision_boundary_original.numpy() == neighbor_class
            )
            # --- Before Unlearning ---
            contour_o = plt.contour(
                xx_o.numpy(), yy_o.numpy(), original_mask,
                colors=['#353535'],  # Jet
                linestyles='dotted',
                linewidths=2,
            )
            
            # Unlearned model boundaries
            unlearned_mask = np.logical_or(
                decision_boundary_unlearned.numpy() == class_idx,
                decision_boundary_unlearned.numpy() == neighbor_class
            )
            # --- After Unlearning ---
            contour_u = plt.contour(
                xx_u.numpy(), yy_u.numpy(), unlearned_mask,
                colors=['#FC9E4F'],  # Carribean Current
                linestyles='solid',
                linewidths=2,
            )
            
    # Get the axis to enhance
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Store the handles for legend
    legend_handles = []
    
    # Add custom legend entries for the two boundaries
    from matplotlib.lines import Line2D    
    original_legend = Line2D([0], [0], color='#353535', lw=2, linestyle='dotted', label='Original')
    unlearned_legend = Line2D([0], [0], color='#FC9E4F', lw=2, linestyle='solid', label='Unlearned')
    legend_handles.extend([original_legend, unlearned_legend])
    
    # Add rogue points with two different styles
    for i, (x_point, rogue_class, rogue_color) in enumerate(zip(X_forget, rogue_classes, rogue_colors)):
        rogue_point = plt.scatter(
            x_point[0], x_point[1], 
            color=rogue_color, 
            marker="X", 
            s=100,
            linewidth=1,
            edgecolor='white',
            zorder=10,  # Higher zorder to ensure visibility
            label="Forget obs." if i == 0 else "_nolegend_"
        )
        if i == 0:
            legend_handles.append(rogue_point)
    
    # Improve title and labels
    plt.title(plot_title if plot_title is not None else "", 
              fontsize=26)
    plt.xlabel('Feature 1', fontsize=22)
    plt.xlabel('Feature 1', fontsize=22)
    plt.ylabel('Feature 2', fontsize=22)
    
    # Add legend with box and all custom entries
    fig.legend(
        handles=legend_handles,
        loc='outside lower center',
        ncol=len(legend_handles),
        frameon=False,
        fontsize=20,
        bbox_to_anchor=(0.55, -0.1),  # Add vertical spacing above legend
        columnspacing=0.5  # Reduce spacing between legend items (default is 2.0)
    )
    
    # Improve ticks
    ax.tick_params(direction='out', length=6, width=1, labelsize=20)
    
    # Add a subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('lightgray')
    
    # Add grid with light alpha
    plt.grid(True, alpha=0.3)
    
    # Ensure tight layout
    # plt.tight_layout() # No longer needed with constrained_layout
    
    
    plt.savefig(save_path, 
                bbox_inches='tight', 
                format='pdf',
                dpi=150)  # Reduced DPI for PDF
    
    plt.close()