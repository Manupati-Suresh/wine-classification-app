# Troubleshooting Guide for Wine Classification App

## 🚨 Common Deployment Issues

### **App Won't Start**
- **Check logs** in Streamlit Cloud dashboard
- **Verify requirements.txt** has all dependencies
- **Ensure app.py** is in the root directory
- **Check Python version** compatibility

### **Model Loading Errors**
- App automatically trains model if missing
- Check if scikit-learn versions match
- Verify pickle file isn't corrupted
- Monitor memory usage (model is ~180KB)

### **Slow Performance**
- **Caching**: Model loads once with @st.cache_resource
- **Memory**: Monitor Streamlit Cloud resource usage
- **Optimization**: Consider model compression if needed

### **UI Issues**
- **Mobile**: App is responsive but test on different devices
- **Browsers**: Test on Chrome, Firefox, Safari
- **Emojis**: Some systems might not display wine emojis

## 🔧 Quick Fixes

### **Update Dependencies**
```bash
# Update requirements.txt with specific versions
streamlit>=1.28.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
```

### **Redeploy App**
1. Make changes to your code
2. Push to GitHub: `git push origin main`
3. Streamlit Cloud auto-redeploys

### **Reset App State**
- Click "Reboot app" in Streamlit Cloud dashboard
- Clear browser cache if UI issues persist

## 📞 Getting Help
- **Streamlit Community**: https://discuss.streamlit.io/
- **GitHub Issues**: Create issues in your repository
- **Documentation**: https://docs.streamlit.io/