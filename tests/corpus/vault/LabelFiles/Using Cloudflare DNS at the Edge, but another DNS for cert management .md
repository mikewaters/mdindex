# Using Cloudflare DNS at the Edge, but another DNS for cert management 

Certainly. I'll prepare a detailed guide for automating Cloudflare DNS with a zone for Certbot managed in another provider. This guide will focus on using Cloudflare as your primary DNS provider while delegating the ACME challenge to another DNS provider (in this case, we'll use Amazon Route 53 as an example, but the concepts can be applied to other providers).

Guide: Automating Cloudflare DNS with Certbot and External ACME Challenge Zone

1. Prerequisites:

   - A domain registered and using Cloudflare for DNS

   - An AWS account with Route 53 access

   - Certbot installed on your system

   - Python and pip installed

2. Set up Cloudflare:

   a. Log in to your Cloudflare account
   b. Select your domain
   c. Go to the DNS settings
   d. Add a new CNAME record:

   - Name: \_acme-challenge

   - Target: [acme-validation.yourdomain.com](http://acme-validation.yourdomain.com)

   - TTL: Auto
      e. Save the record

3. Set up Route 53:

   a. Log in to AWS Console
   b. Go to Route 53
   c. Create a new hosted zone for "[yourdomain.com](http://yourdomain.com)"
   d. Note the nameservers for this zone (you won't use these for your main domain)

4. Install required Python packages:

   ```
   pip install certbot certbot-dns-cloudflare certbot-dns-route53
   ```

5. Configure Cloudflare API credentials:

   a. Create a file named cloudflare.ini:

   ```
   dns_cloudflare_email = your-cloudflare-email@example.com
   dns_cloudflare_api_key = your-global-api-key
   ```

   b. Secure this file:

   ```
   chmod 600 cloudflare.ini
   ```

6. Configure AWS credentials:

   a. Create a file named aws_credentials:

   ```
   [default]
   aws_access_key_id = YOUR_ACCESS_KEY
   aws_secret_access_key = YOUR_SECRET_KEY
   ```

   b. Secure this file:

   ```
   chmod 600 aws_credentials
   ```

7. Create a script to automate the certificate issuance/renewal:

   Create a file named `renew_cert.sh`:

   ```bash
   #!/bin/bash
   
   # Renew the certificate
   certbot certonly \
     --dns-cloudflare \
     --dns-cloudflare-credentials ./cloudflare.ini \
     --dns-cloudflare-propagation-seconds 60 \
     --cert-name yourdomain.com \
     --domains yourdomain.com,*.yourdomain.com \
     --preferred-challenges dns-01 \
     --manual-auth-hook ./auth_hook.sh \
     --manual-cleanup-hook ./cleanup_hook.sh \
     --force-renewal
   
   # Restart web server (adjust as needed)
   systemctl restart nginx
   ```

8. Create the auth hook script:

   Create a file named `auth_hook.sh`:

   ```bash
   #!/bin/bash
   
   # Load AWS credentials
   export AWS_SHARED_CREDENTIALS_FILE="./aws_credentials"
   
   # Extract the domain and validation string
   DOMAIN=$(echo $CERTBOT_DOMAIN | cut -d'.' -f1)
   VALIDATION="$CERTBOT_VALIDATION"
   
   # Use AWS CLI to add the TXT record
   aws route53 change-resource-record-sets \
     --hosted-zone-id YOUR_ROUTE53_ZONE_ID \
     --change-batch '{
       "Changes": [{
         "Action": "UPSERT",
         "ResourceRecordSet": {
           "Name": "acme-validation.'$DOMAIN'.com",
           "Type": "TXT",
           "TTL": 60,
           "ResourceRecords": [{"Value": "\"'$VALIDATION'\""}]
         }
       }]
     }'
   ```

9. Create the cleanup hook script:

   Create a file named `cleanup_hook.sh`:

   ```bash
   #!/bin/bash
   
   # Load AWS credentials
   export AWS_SHARED_CREDENTIALS_FILE="./aws_credentials"
   
   # Extract the domain
   DOMAIN=$(echo $CERTBOT_DOMAIN | cut -d'.' -f1)
   
   # Use AWS CLI to remove the TXT record
   aws route53 change-resource-record-sets \
     --hosted-zone-id YOUR_ROUTE53_ZONE_ID \
     --change-batch '{
       "Changes": [{
         "Action": "DELETE",
         "ResourceRecordSet": {
           "Name": "acme-validation.'$DOMAIN'.com",
           "Type": "TXT",
           "TTL": 60,
           "ResourceRecords": [{"Value": "\"'$CERTBOT_VALIDATION'\""}]
         }
       }]
     }'
   ```

10. Make the scripts executable:

   ```
   chmod +x renew_cert.sh auth_hook.sh cleanup_hook.sh
   ```

11. Test the renewal process:

   ```
   ./renew_cert.sh
   ```

12. Set up a cron job for automatic renewal:

   Edit your crontab:

   ```
   crontab -e
   ```

   Add this line to run the renewal script daily:

   ```
   0 0 * * * /path/to/renew_cert.sh >> /var/log/certbot-renewal.log 2>&1
   ```

This setup accomplishes the following:

- Uses Cloudflare as the primary DNS provider

- Delegates the ACME challenge to a Route 53 zone

- Automates the certificate issuance and renewal process

- Provides flexibility to use different providers for main DNS and ACME challenges

Key points to remember:

1. Replace placeholders (YOUR_ROUTE53_ZONE_ID, [yourdomain.com](http://yourdomain.com), etc.) with your actual values.

2. Ensure your AWS IAM user has appropriate permissions for Route 53.

3. Keep your API credentials and key files secure.

4. Test thoroughly before relying on this for production certificates.

5. Monitor the renewal logs to ensure the process is working correctly.

This setup provides a robust and flexible solution for managing SSL/TLS certificates with Cloudflare DNS while leveraging another provider for ACME challenges. It allows you to maintain your primary DNS with Cloudflare while taking advantage of the automation capabilities of other providers for certificate management.​​​​​​​​​​​​​​​​