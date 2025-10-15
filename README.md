# 🏗️ RC Beam Designer - ACI 318

Professional web application for designing reinforced concrete beams according to ACI 318 Building Code Requirements.

## 📋 Features

- ✅ **Full ACI 318-19 Compliance**: Rectangular stress block method
- 📊 **Interactive Design**: Real-time calculations and visualizations
- 🎨 **Professional UI**: Modern, intuitive interface
- 📈 **Visual Analysis**: Cross-section and strain diagrams
- 💾 **Export Results**: CSV and text file downloads
- 🔍 **Design Verification**: Automatic code compliance checks
- 📚 **Built-in Help**: Educational content and design guides

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or download the project**
```bash
# Create project directory
mkdir rc_beam_designer
cd rc_beam_designer
```

2. **Create virtual environment (recommended)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open in browser**
```
The app will automatically open at http://localhost:8501
```

## 📁 Project Structure

```
rc_beam_designer/
│
├── app.py                  # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
└── (optional folders)
    ├── exports/           # For saved results
    ├── docs/              # Additional documentation
    └── examples/          # Example calculations
```

## 📖 How to Use

### Step 1: Input Material Properties
- **Concrete Strength (f'c)**: 20-80 MPa
- **Steel Yield Strength (fy)**: 300-600 MPa

### Step 2: Define Beam Geometry
- **Width (b)**: 200-1000 mm
- **Height (h)**: 300-1500 mm
- **Cover**: 20-75 mm

### Step 3: Enter Design Loads
- **Factored Moment (Mu)**: In kN·m

### Step 4: Select Reinforcement
- **Bar Diameter**: ø10 to ø32 mm
- **Stirrup Size**: ø8 to ø12 mm

### Step 5: Design & Analyze
- Click **"Design Beam"** button
- Review results and visualizations
- Select bar arrangement
- Export results

## 🎓 Design Method

### ACI 318-19 Provisions

The application implements:

- **Section 22.2**: Flexural strength calculations
- **Section 9.6**: Minimum reinforcement requirements
- **Section 21.2**: Strength reduction factors (φ)

### Key Equations

```
Required Steel Area:
As = ρ × b × d

Moment Capacity:
Mn = As × fy × (d - a/2)

Compression Block Depth:
a = As × fy / (0.85 × f'c × b)

Strength Reduction Factor:
φ = 0.65 to 0.90 (based on strain limits)
```

### Design Checks

✅ Minimum reinforcement ratio (ρ_min)  
✅ Maximum reinforcement ratio (ρ_max = 0.75ρ_b)  
✅ Strain limits (tension/compression control)  
✅ Moment capacity adequacy (φMn ≥ Mu)  

## 📊 Output

The application provides:

1. **Detailed Results Table**
   - Required steel area
   - Reinforcement ratios
   - Neutral axis depth
   - Strain analysis
   - Moment capacities

2. **Bar Arrangement Suggestions**
   - Multiple practical options
   - Area comparisons
   - Excess percentages

3. **Visual Diagrams**
   - Beam cross-section with reinforcement
   - Strain distribution diagram

4. **Export Options**
   - CSV results file
   - Text summary report

## ⚙️ Configuration

### Customization Options

Edit `app.py` to customize:

```python
# Default values
DEFAULT_FC = 25.0  # MPa
DEFAULT_FY = 420.0  # MPa
DEFAULT_WIDTH = 300.0  # mm
DEFAULT_HEIGHT = 600.0  # mm

# Available bar sizes
BAR_SIZES = [10, 12, 16, 20, 25, 28, 32]  # mm

# Stirrup sizes
STIRRUP_SIZES = [8, 10, 12]  # mm
```

## 🔧 Advanced Features (Optional)

To enable additional features, uncomment in `requirements.txt` and implement:

### PDF Export
```bash
pip install reportlab
```

### Excel Export
```bash
pip install openpyxl xlsxwriter
```

### Database Storage
```bash
pip install sqlalchemy
```

## ⚠️ Important Notes

### Limitations

- ✋ Flexural design only (shear design not included)
- ✋ Singly reinforced rectangular sections only
- ✋ Serviceability checks not implemented
- ✋ Development length must be verified separately

### Assumptions

- Plane sections remain plane
- Perfect bond between concrete and steel
- Concrete tensile strength neglected
- Linear strain distribution
- Maximum concrete strain = 0.003

### Recommendations

1. ✅ Verify results with hand calculations
2. ✅ Check local building codes
3. ✅ Consider constructability
4. ✅ Verify bar spacing requirements
5. ✅ Perform separate shear design
6. ✅ Check development length
7. ✅ Consider deflection limits

## 🐛 Troubleshooting

### Common Issues

**Problem**: App won't start
```bash
# Solution: Check Python version
python --version  # Should be 3.8+

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**Problem**: Import errors
```bash
# Solution: Activate virtual environment
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
```

**Problem**: Plots not displaying
```bash
# Solution: Update matplotlib
pip install --upgrade matplotlib
```

## 📚 References

- **ACI 318-19**: Building Code Requirements for Structural Concrete
- **ACI 318.1-19**: Building Code Requirements for Structural Concrete Commentary
- MacGregor, J.G., & Wight, J.K. (2016). *Reinforced Concrete: Mechanics and Design*

## 📝 License

This is an educational tool. For production use:
- Verify all calculations with licensed professional engineer
- Follow local building codes and regulations
- Perform comprehensive structural analysis

## 👨‍💻 Development

### Adding New Features

1. **Doubly Reinforced Beams**
   - Add compression steel input
   - Modify calculation functions
   - Update visualization

2. **Shear Design**
   - Add shear force input
   - Implement stirrup design
   - Calculate spacing

3. **T-Beam Analysis**
   - Add flange dimensions
   - Implement effective width
   - Modify stress block

4. **Load Combinations**
   - Add multiple load cases
   - Implement load factors
   - Show envelope

### Testing

```bash
# Run with different inputs
# Verify against hand calculations
# Check edge cases (min/max values)
# Test export functions
```

## 🤝 Contributing

Suggestions for improvement:
- Additional design features
- Bug fixes
- UI enhancements
- Documentation improvements

## 📞 Support

For questions or issues:
- Check the built-in help section
- Review ACI 318 documentation
- Consult with structural engineer

## 🔄 Version History

**v1.0.0** (Current)
- Initial release
- ACI 318-19 flexural design
- Interactive UI with visualizations
- Export functionality

---

**Built with ❤️ using Streamlit**

*Designed for educational and preliminary design purposes*
