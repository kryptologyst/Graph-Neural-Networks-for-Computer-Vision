"""Streamlit demo for Graph Neural Networks for Computer Vision."""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.device import get_device, set_seed
from src.data.dataset import SuperpixelGraphDataset
from src.models.gnn_models import create_model
from src.eval.metrics import compute_metrics, plot_confusion_matrix, analyze_predictions


def load_model_and_data(model_type: str, device: torch.device):
    """Load model and data for inference."""
    config = Config()
    config.model.model_type = model_type
    
    # Load dataset
    dataset = SuperpixelGraphDataset(
        dataset_name=config.data.dataset_name,
        data_root=config.data.data_root,
        superpixel_segments=config.data.superpixel_segments,
        superpixel_compactness=config.data.superpixel_compactness,
        normalize=config.data.normalize
    )
    
    # Create model
    model = create_model(
        model_type=config.model.model_type,
        in_channels=dataset.num_node_features,
        hidden_channels=config.model.hidden_channels,
        num_classes=dataset.num_classes,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        activation=config.model.activation,
        use_batch_norm=config.model.use_batch_norm,
        use_residual=config.model.use_residual,
        pooling=config.model.pooling
    )
    
    # Load checkpoint if available
    checkpoint_path = f"checkpoints/best_model_{model_type}.pt"
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    
    return model, dataset


def visualize_superpixel_graph(data, sample_idx: int = 0):
    """Visualize a superpixel graph."""
    # Get the graph data
    x = data[sample_idx].x.numpy()
    edge_index = data[sample_idx].edge_index.numpy()
    y = data[sample_idx].y.item()
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes with features
    for i, features in enumerate(x):
        G.add_node(i, features=features)
    
    # Add edges
    for src, dst in edge_index.T:
        G.add_edge(src, dst)
    
    # Create layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Edges'
    ))
    
    # Add nodes
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'Node {node}<br>Features: {G.nodes[node]["features"]}')
        # Color nodes by RGB features
        rgb = G.nodes[node]["features"][:3]  # Take first 3 features as RGB
        node_colors.append(f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})')
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=10,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        name='Nodes'
    ))
    
    fig.update_layout(
        title=f'Superpixel Graph Visualization (Class: {y})',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Interactive superpixel graph - hover over nodes to see features",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color='gray', size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="GNN for Computer Vision",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Graph Neural Networks for Computer Vision")
    st.markdown("""
    This demo showcases Graph Neural Networks applied to computer vision tasks using superpixel-based graph representations.
    Images are converted into graphs where superpixels become nodes and spatial relationships become edges.
    """)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["gcn", "graphsage", "gat", "gin", "transformer"],
        index=0
    )
    
    # Device selection
    device_option = st.sidebar.selectbox(
        "Device",
        ["auto", "cpu", "cuda", "mps"],
        index=0
    )
    
    device = get_device(device_option)
    st.sidebar.info(f"Using device: {device}")
    
    # Load model and data
    try:
        with st.spinner("Loading model and data..."):
            model, dataset = load_model_and_data(model_type, device)
            model = model.to(device)
            
            # Get data loaders
            train_loader, val_loader, test_loader = dataset.get_data_loaders(
                train_size=100, val_size=50, test_size=50,
                batch_size=1, num_workers=0
            )
            
        st.success(f"Loaded {model_type.upper()} model successfully!")
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please train a model first using the training script.")
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "🔍 Graph Visualization", "🎯 Model Predictions", "📈 Model Analysis"])
    
    with tab1:
        st.header("Dataset Overview")
        
        # Dataset info
        dataset_info = dataset.get_dataset_info()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", dataset_info['total_samples'])
        with col2:
            st.metric("Number of Classes", dataset_info['num_classes'])
        with col3:
            st.metric("Node Features", dataset_info['num_node_features'])
        with col4:
            st.metric("Superpixel Segments", dataset_info['superpixel_segments'])
        
        # Model info
        st.subheader("Model Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Type", model_type.upper())
            st.metric("Hidden Channels", dataset_info['num_node_features'])
        with col2:
            st.metric("Parameters", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            st.metric("Model Size", f"{sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2:.2f} MB")
    
    with tab2:
        st.header("Superpixel Graph Visualization")
        
        # Sample selection
        sample_idx = st.slider("Select Sample", 0, min(49, len(test_loader.dataset)-1), 0)
        
        # Get sample data
        sample_data = test_loader.dataset[sample_idx]
        
        # Visualize graph
        fig = visualize_superpixel_graph(test_loader.dataset, sample_idx)
        st.plotly_chart(fig, use_container_width=True)
        
        # Graph statistics
        st.subheader("Graph Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Nodes", sample_data.num_nodes)
        with col2:
            st.metric("Number of Edges", sample_data.edge_index.size(1))
        with col3:
            st.metric("True Class", sample_data.y.item())
        
        # Node features distribution
        st.subheader("Node Features Distribution")
        features_df = pd.DataFrame(sample_data.x.numpy(), columns=[f'Feature {i}' for i in range(sample_data.x.size(1))])
        st.dataframe(features_df.describe())
    
    with tab3:
        st.header("Model Predictions")
        
        # Sample selection for prediction
        pred_sample_idx = st.slider("Select Sample for Prediction", 0, min(49, len(test_loader.dataset)-1), 0)
        
        # Get prediction
        with torch.no_grad():
            sample = test_loader.dataset[pred_sample_idx].to(device)
            logits = model(sample.x, sample.edge_index, torch.zeros(sample.num_nodes, dtype=torch.long, device=device))
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(logits, dim=1).item()
            confidence = torch.max(probs).item()
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("True Class", sample.y.item())
        with col2:
            st.metric("Predicted Class", pred_class)
        with col3:
            st.metric("Confidence", f"{confidence:.3f}")
        
        # Prediction probabilities
        st.subheader("Class Probabilities")
        class_probs = probs.cpu().numpy()[0]
        
        fig = px.bar(
            x=list(range(len(class_probs))),
            y=class_probs,
            title="Prediction Probabilities",
            labels={'x': 'Class', 'y': 'Probability'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correctness indicator
        is_correct = pred_class == sample.y.item()
        if is_correct:
            st.success("✅ Prediction is correct!")
        else:
            st.error("❌ Prediction is incorrect!")
    
    with tab4:
        st.header("Model Analysis")
        
        # Evaluate on test set
        if st.button("Evaluate Model on Test Set"):
            with st.spinner("Evaluating model..."):
                all_preds = []
                all_labels = []
                all_probs = []
                
                model.eval()
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch.to(device)
                        logits = model(batch.x, batch.edge_index, batch.batch)
                        probs = torch.softmax(logits, dim=1)
                        preds = torch.argmax(logits, dim=1)
                        
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(batch.y.cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())
                
                # Compute metrics
                metrics = compute_metrics(all_labels, all_preds, ["accuracy", "f1_macro", "f1_micro"])
                
                # Display metrics
                st.subheader("Test Set Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                with col2:
                    st.metric("F1 Macro", f"{metrics['f1_macro']:.3f}")
                with col3:
                    st.metric("F1 Micro", f"{metrics['f1_micro']:.3f}")
                
                # Confusion matrix
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_confusion_matrix(all_labels, all_preds, save_path=None)
                st.pyplot(fig)
                
                # Analysis
                analysis = analyze_predictions(all_labels, all_preds, all_probs)
                st.subheader("Prediction Analysis")
                st.json(analysis)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this demo:**
    - Uses superpixel-based graph representations of images
    - Implements multiple GNN architectures (GCN, GraphSAGE, GAT, GIN, Transformer)
    - Provides interactive visualization and analysis tools
    - Built with PyTorch Geometric and Streamlit
    """)


if __name__ == "__main__":
    import pandas as pd
    main()
