def generate_tooltip_html(description):
    tooltip_html = f"""
    <style>
    .tooltip {{
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted black;
    }}

    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 200px;
        background-color: black;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px 0;
        position: absolute;
        z-index: 1;
        bottom: 125%; /* Position the tooltip above the text */
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }}

    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    </style>

    <div class="tooltip">ℹ️
        <span class="tooltiptext">{description}</span>
    </div>
    """
    return tooltip_html 