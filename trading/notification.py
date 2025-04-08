import smtplib
from email.mime.text import MIMEText
import requests

def send_email_notification(message, recipient):
    """
    发送邮件通知。

    Args:
        message (str): 通知内容。
        recipient (str): 接收者的邮件地址。
    """
    try:
        sender_email = "your_email@example.com"  # 替换为实际发送者邮箱
        sender_password = "your_password"  # 替换为实际密码
        smtp_server = "smtp.example.com"  # 替换为实际SMTP服务器
        smtp_port = 587  # 替换为实际端口

        msg = MIMEText(message)
        msg['Subject'] = "AI-Trader Notification"
        msg['From'] = sender_email
        msg['To'] = recipient

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient, msg.as_string())
        print(f"Email notification sent to {recipient}.")
    except Exception as e:
        print(f"Failed to send email notification: {e}")