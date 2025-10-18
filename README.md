
# WhatsApp Chatbot - Hosted on Railway

Deze bot gebruikt een zelfgetraind taalmodel en draait 24/7 op Railway via een Flask-server.

## 🚀 Deployment stappen

1. Upload deze bestanden naar een **nieuwe GitHub repo**
2. Deploy de repo op [Railway](https://railway.app)
3. Voeg de variabelen toe via het `.env.template` bestand
4. Stel je Twilio webhook in:
   - Ga naar je Twilio Console → WhatsApp sender → **Webhook URL**
   - Gebruik je Railway URL, bijvoorbeeld:
     ```
     https://jouw-app.up.railway.app/whatsapp
     ```

## 🧠 Model
Het model wordt automatisch gedownload van Google Drive bij de eerste start.

## ✅ .env template

```env
TWILIO_ACCOUNT_SID=US42685bad8695b7ba025418dfddf28f0c
TWILIO_AUTH_TOKEN= ...
TWILIO_WHATSAPP_NUMBER=+15558665761
YOUR_PHONE_NUMBER=+31687437563
```

