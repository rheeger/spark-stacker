"""
Visualization module for generating charts for backtesting reports.
Provides functions to create interactive plots using Plotly.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_price_chart(
    df: pd.DataFrame,
    trades: List[Dict],
    filename: str,
    show_indicators: bool = True
) -> str:
    """
    Generate an interactive price chart with trade entries/exits marked.

    Args:
        df: DataFrame containing OHLCV data with any indicator columns
        trades: List of trade dictionaries with entry/exit timestamps and prices
        filename: Output filename for the chart
        show_indicators: Whether to display indicator values on the chart

    Returns:
        Path to the saved HTML chart file
    """
    try:
        # Ensure we have the required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column {col} not found in DataFrame")

        # Determine which columns might be indicators (not in the required list)
        indicator_cols = [col for col in df.columns if col not in required_cols]

        # Create subplots: price chart and volume
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price", "Volume")
        )

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price"
            ),
            row=1, col=1
        )

        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['volume'],
                name="Volume",
                marker_color='rgba(0, 0, 255, 0.3)'
            ),
            row=2, col=1
        )

        # Add indicators if requested
        if show_indicators and indicator_cols:
            for i, indicator in enumerate(indicator_cols):
                if indicator in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=df[indicator],
                            mode='lines',
                            name=indicator,
                            line=dict(width=1)
                        ),
                        row=1, col=1
                    )

        # Add trade entries and exits
        # Track whether we've already added legend entries
        entry_legend_added = False
        exit_legend_added = False

        for trade in trades:
            # Entry point
            if 'entry_time' in trade and 'entry_price' in trade:
                # Handle timestamps properly - could be milliseconds or already datetime
                entry_time = trade['entry_time']
                if isinstance(entry_time, (int, float)):
                    entry_datetime = pd.to_datetime(entry_time, unit='ms')
                else:
                    entry_datetime = pd.to_datetime(entry_time)

                fig.add_trace(
                    go.Scatter(
                        x=[entry_datetime],
                        y=[trade['entry_price']],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',  # Use triangle-up for all entries
                            size=10,
                            color='blue',  # Keep blue for entry markers
                        ),
                        name='Entry',
                        showlegend=not entry_legend_added,  # Only show legend for first entry
                        legendgroup='entries'  # Group all entries together
                    ),
                    row=1, col=1
                )
                entry_legend_added = True

            # Exit point
            if 'exit_time' in trade and 'exit_price' in trade:
                # Handle timestamps properly - could be milliseconds or already datetime
                exit_time = trade['exit_time']
                if isinstance(exit_time, (int, float)):
                    exit_datetime = pd.to_datetime(exit_time, unit='ms')
                else:
                    exit_datetime = pd.to_datetime(exit_time)

                fig.add_trace(
                    go.Scatter(
                        x=[exit_datetime],
                        y=[trade['exit_price']],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=10,
                            color='purple',  # Changed to purple for better visibility
                        ),
                        name='Exit',
                        showlegend=not exit_legend_added,  # Only show legend for first exit
                        legendgroup='exits'  # Group all exits together
                    ),
                    row=1, col=1
                )
                exit_legend_added = True

        # Update layout for better visualization
        fig.update_layout(
            title="Price Chart with Trades",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            autosize=True,
            legend_title="Legend",
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            margin=dict(l=50, r=50, t=50, b=50)
        )

        # Save as HTML file with responsive configuration
        output_path = Path(filename)
        config = {
            'displayModeBar': True,
            'responsive': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
        }
        fig.write_html(
            output_path,
            config=config,
            include_plotlyjs='cdn'
        )
        logger.info(f"Generated price chart: {output_path}")

        return str(output_path)

    except Exception as e:
        logger.error(f"Error generating price chart: {e}")
        raise


def generate_equity_curve(
    trades: List[Dict],
    filename: str,
    initial_capital: float = 10000.0
) -> Tuple[str, pd.DataFrame]:
    """
    Generate an equity curve chart from a list of trades.

    Args:
        trades: List of trade dictionaries with P&L information
        filename: Output filename for the chart
        initial_capital: Starting capital amount

    Returns:
        Tuple of (path to the saved HTML chart file, equity curve DataFrame)
    """
    try:
        # Validate trades data
        if not trades:
            logger.warning("No trades provided for equity curve generation")
            return None, pd.DataFrame()

        # Create DataFrame with dates and P&L
        trade_data = []
        for trade in trades:
            # Make sure we have the minimum required data
            if 'exit_time' not in trade or 'pnl' not in trade:
                logger.warning(f"Trade missing required fields: {trade}")
                continue

            # Handle timestamps properly - could be milliseconds or already datetime
            exit_time = trade['exit_time']
            if isinstance(exit_time, (int, float)):
                exit_datetime = pd.to_datetime(exit_time, unit='ms')
            else:
                exit_datetime = pd.to_datetime(exit_time)

            trade_data.append({
                'date': exit_datetime,
                'pnl': float(trade['pnl']),
                'cumulative_pnl': 0.0  # Will be calculated below
            })

        if not trade_data:
            logger.warning("No valid trades with required fields found")
            return None, pd.DataFrame()

        # Convert to DataFrame and sort by date
        equity_df = pd.DataFrame(trade_data)
        equity_df = equity_df.sort_values('date')

        # Calculate cumulative P&L and equity as percentage of initial capital
        equity_df['cumulative_pnl'] = equity_df['pnl'].cumsum()
        equity_df['equity_absolute'] = initial_capital + equity_df['cumulative_pnl']
        equity_df['equity_percentage'] = (equity_df['equity_absolute'] / initial_capital) * 100

        # Add maximum drawdown calculation using percentage equity
        equity_df['peak_percentage'] = equity_df['equity_percentage'].cummax()
        equity_df['drawdown'] = ((equity_df['equity_percentage'] - equity_df['peak_percentage']) / equity_df['peak_percentage']) * 100

        # For backward compatibility, keep the absolute equity column as 'equity'
        equity_df['equity'] = equity_df['equity_absolute']

        # Create the equity curve plot
        fig = go.Figure()

        # Add equity curve line using percentage values
        fig.add_trace(
            go.Scatter(
                x=equity_df['date'],
                y=equity_df['equity_percentage'],
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2),
                hovertemplate='Date: %{x}<br>Account Value: %{y:.2f}%<extra></extra>'
            )
        )

        # Add horizontal line for initial capital at 100%
        fig.add_hline(
            y=100,
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial Capital: 100%"
        )

        # Update layout
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Account Value (%)",
            height=600,
            autosize=True,
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=50, r=50, t=50, b=50),
            yaxis=dict(
                tickformat=".1f",
                ticksuffix="%",
                # Always start from 0% (complete loss) and scale up as needed
                range=[
                    0,  # Always start from 0% (complete loss)
                    max(120, equity_df['equity_percentage'].max() + 10)  # Scale up to at least 120% or 10% above max
                ],
                dtick=10  # Show tick marks every 10%
            )
        )

        # Save as HTML file with responsive configuration
        output_path = Path(filename)
        config = {
            'displayModeBar': True,
            'responsive': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
        }
        fig.write_html(
            output_path,
            config=config,
            include_plotlyjs='cdn'
        )
        logger.info(f"Generated equity curve: {output_path}")

        return str(output_path), equity_df

    except Exception as e:
        logger.error(f"Error generating equity curve: {e}")
        raise


def generate_drawdown_chart(
    equity_curve: pd.DataFrame,
    filename: str
) -> str:
    """
    Generate a drawdown chart from an equity curve.

    Args:
        equity_curve: DataFrame with equity and drawdown data
        filename: Output filename for the chart

    Returns:
        Path to the saved HTML chart file
    """
    try:
        # Validate input
        required_cols = ['date', 'drawdown']
        for col in required_cols:
            if col not in equity_curve.columns:
                raise ValueError(f"Required column {col} not found in equity curve DataFrame")

        # Create the drawdown plot
        fig = go.Figure()

        # Add drawdown line
        fig.add_trace(
            go.Scatter(
                x=equity_curve['date'],
                y=equity_curve['drawdown'],
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=2),
                fill='tozeroy',  # Fill area to the x-axis
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
            )
        )

        # Calculate and mark maximum drawdown
        max_drawdown_idx = equity_curve['drawdown'].idxmin()
        if max_drawdown_idx is not None:
            max_drawdown_value = equity_curve.loc[max_drawdown_idx, 'drawdown']
            max_drawdown_date = equity_curve.loc[max_drawdown_idx, 'date']

            fig.add_trace(
                go.Scatter(
                    x=[max_drawdown_date],
                    y=[max_drawdown_value],
                    mode='markers+text',
                    marker=dict(size=10, color='black'),
                    text=[f"{max_drawdown_value:.2f}%"],
                    textposition="bottom center",
                    name='Max Drawdown',
                    hovertemplate='Date: %{x}<br>Max Drawdown: %{y:.2f}%<extra></extra>'
                )
            )

            # Add annotation for maximum drawdown
            fig.add_annotation(
                x=max_drawdown_date,
                y=max_drawdown_value,
                text=f"Max DD: {max_drawdown_value:.2f}%",
                showarrow=True,
                arrowhead=1,
                arrowcolor="black",
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )

        # Add horizontal line at 0%
        fig.add_hline(
            y=0,
            line_dash="solid",
            line_color="gray",
            annotation_text="Break-even: 0%"
        )

        # Update layout
        fig.update_layout(
            title="Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=500,
            autosize=True,
            template="plotly_white",
            yaxis=dict(
                tickformat=".1f",
                ticksuffix="%",
                # Always start from -100% (complete loss) and scale up
                range=[
                    -100,  # Always start from -100% (complete loss)
                    5  # End at +5% to show clearly above the zero line
                ],
                dtick=10  # Show tick marks every 10%
            ),
            hovermode="x unified",
            margin=dict(l=50, r=50, t=50, b=50)
        )

        # Save as HTML file with responsive configuration
        output_path = Path(filename)
        config = {
            'displayModeBar': True,
            'responsive': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
        }
        fig.write_html(
            output_path,
            config=config,
            include_plotlyjs='cdn'
        )
        logger.info(f"Generated drawdown chart: {output_path}")

        return str(output_path)

    except Exception as e:
        logger.error(f"Error generating drawdown chart: {e}")
        raise
