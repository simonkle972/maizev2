"""
Seed institutions from the Hipo university-domains-list dataset.
https://github.com/Hipo/university-domains-list

Usage:
    export DOTENV_PATH=.env.local
    python seed_institutions.py

Idempotent: safe to run multiple times. Existing institutions matched by
(name, alpha_two_code) are updated in place; their domains are reconciled.
Also migrates any existing Institution.email_domain values into the
institution_domains table for backward compat.
"""

import os
import sys
import urllib.request
import json

# Load env before importing app
dotenv_path = os.environ.get('DOTENV_PATH', '.env.local')
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path, override=True)
except ImportError:
    pass

from app import app
from models import db, Institution, InstitutionDomain

DATASET_URL = (
    "https://raw.githubusercontent.com/Hipo/university-domains-list/"
    "master/world_universities_and_domains.json"
)


def fetch_dataset():
    print(f"Downloading dataset from {DATASET_URL} ...")
    with urllib.request.urlopen(DATASET_URL, timeout=60) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    print(f"  Downloaded {len(data)} university records.")
    return data


def seed(data):
    institutions_upserted = 0
    domains_inserted = 0

    for entry in data:
        name = (entry.get('name') or '').strip()
        if not name:
            continue

        alpha_two = (entry.get('alpha_two_code') or '').strip().upper() or None
        country = (entry.get('country') or '').strip() or None
        state_province = (entry.get('state-province') or '').strip() or None
        web_pages = entry.get('web_pages') or []
        raw_domains = [d.strip().lower() for d in (entry.get('domains') or []) if d.strip()]

        # Upsert institution (match on name + alpha_two_code)
        institution = None
        if alpha_two:
            institution = Institution.query.filter_by(
                name=name, alpha_two_code=alpha_two
            ).first()
        if institution is None:
            institution = Institution.query.filter_by(
                name=name, alpha_two_code=None
            ).first()
        if institution is None:
            institution = Institution(name=name)
            db.session.add(institution)
            db.session.flush()

        institution.is_from_dataset = True
        institution.country = country
        institution.alpha_two_code = alpha_two
        institution.state_province = state_province
        institution.web_pages = web_pages
        institutions_upserted += 1

        # Reconcile domains
        existing_domains = {d.domain for d in institution.domains}
        for domain in raw_domains:
            if domain not in existing_domains:
                db.session.add(InstitutionDomain(
                    institution_id=institution.id,
                    domain=domain
                ))
                domains_inserted += 1

        if institutions_upserted % 500 == 0:
            db.session.flush()
            print(f"  ... {institutions_upserted} institutions processed")

    db.session.flush()
    return institutions_upserted, domains_inserted


def migrate_legacy_email_domains():
    """Move existing Institution.email_domain values into institution_domains."""
    migrated = 0
    for inst in Institution.query.filter(
        Institution.email_domain.isnot(None)
    ).all():
        domain = inst.email_domain.strip().lower()
        if not domain:
            continue
        already_exists = InstitutionDomain.query.filter_by(
            institution_id=inst.id, domain=domain
        ).first()
        if not already_exists:
            db.session.add(InstitutionDomain(
                institution_id=inst.id,
                domain=domain
            ))
            migrated += 1
    return migrated


def main():
    with app.app_context():
        try:
            data = fetch_dataset()
        except Exception as e:
            print(f"ERROR: Could not download dataset: {e}")
            sys.exit(1)

        print("Seeding institutions ...")
        upserted, domains = seed(data)

        print("Migrating legacy email_domain fields ...")
        legacy = migrate_legacy_email_domains()

        db.session.commit()
        print()
        print(f"Done.")
        print(f"  Institutions upserted : {upserted}")
        print(f"  Domains inserted      : {domains}")
        print(f"  Legacy domains migrated: {legacy}")


if __name__ == '__main__':
    main()
