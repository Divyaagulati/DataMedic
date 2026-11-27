from email.utils import parsedate_to_datetime

from datamedic.email import OutgoingEmail, build_message


def test_message_has_date_and_message_id_on_the_sender_domain() -> None:
    msg = build_message(
        "DataMedic <sender@mail.example.org>",
        OutgoingEmail(to="user@example.com", subject="Verify", body="link\n"),
    )
    assert msg["To"] == "user@example.com" and msg["Subject"] == "Verify"
    assert parsedate_to_datetime(msg["Date"]).tzinfo is not None
    assert msg["Message-ID"].startswith("<") and msg["Message-ID"].endswith("@mail.example.org>")
    assert msg["Reply-To"] == "DataMedic <sender@mail.example.org>"
    assert msg.get_content() == "link\n"
